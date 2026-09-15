from enum import Enum, auto
from typing import Any, List, Tuple, Set, Optional
from collections import deque
# add local imports
from .vocabulary import VocabularyManager
from .schemas import FunctionDefinition, ParameterProperty
from .errors import CallMeError


class State(Enum):
    """ Enumerations for each possible generation state to further determine if
            LLM should be called or bypassed in a given state.
    """
    # FIXME: I think each "function call" should be wrapped in []
    # --- FORCED / DETERMINISTIC STATES (LLM Bypassed) ---
    EMIT_START = auto()          # Emits '{"name": "'
    EMIT_PARAMS_HEADER = auto()  # Emits '", "parameters": {'
    EMIT_PARAM_KEY = auto()      # Emits '"<param_name>": '
    EMIT_PARAM_SEP = auto()      # Emits ', ' (when moving to the next parameter)
    EMIT_END = auto()            # Emits '}}'

    # --- LLM-DRIVEN STATES (LLM Called) ---
    SELECT_FUNCTION = auto()     # LLM picks function name (e.g., "fn_add_numbers")
    GEN_STRING = auto()          # LLM generates string argument content
    GEN_NUMBER = auto()          # LLM generates numeric argument digits/decimal
    GEN_BOOLEAN = auto()         # LLM selects 'true' or 'false'

    # --- TERMINAL STATE ---
    END = auto()                 # Generation finished


class JSONStateMachine:
    """Tracks current state during token generation and determines allowed
              next tokens.
      """
    def __init__(
            self,
            # prompt_txt: str,
            functions: List[FunctionDefinition],
            vocab_mgr: VocabularyManager
    )-> None:
        """Initialize the state machine with prompt, functions, and vocabulary.

          Args:
              # prompt_txt: The natural language user prompt.
              funcs: List of available function definitions.
              vocab_mgr: VocabularyManager instance for token lookups.
          """
        # self.prompt_txt = prompt_txt
        # FIXME: actually remove prompt_txt or uncomment if we end up using it
        self.functions = functions
        self.vocab_mgr = vocab_mgr

        self.current_state = State.EMIT_START
        self.selected_function: Optional[FunctionDefinition] = None
        self.parameter_queue: deque[Tuple[str, ParameterProperty]] = deque()
        self._param_has_content = False
        self.fn_name_buffer = ""
        self.param_value_buffer = ""
        self.buffer = ""
    
    def _commit_param_value(self) -> None:
        """Appends accumulated parameter value to main buffer and resets local
                buffer.
        """
        self.buffer += self.param_value_buffer
        self.param_value_buffer = ""

    def advance_deterministic(self) -> bool:
        """Bypasses LLM by appending mandatory syntax tokens directly. 
                Returns True if state changed or False otherwise.
        """
        if self.current_state == State.EMIT_START:
            self.buffer += '{"name": "'
            self.current_state = State.SELECT_FUNCTION
            return True
        
        if self.current_state == State.EMIT_PARAMS_HEADER:
            self.buffer += ', "parameters": {'
            self.current_state = State.EMIT_PARAM_KEY if self.parameter_queue \
                    else State.EMIT_END
            return True
       
        if self.current_state == State.EMIT_PARAM_KEY:
            p_name, p_prop = self.parameter_queue.popleft()
            self._param_has_content = False
            self.param_value_buffer = ""
            if p_prop.type == "string":
                self.buffer += f'"{p_name}": "'
                self.current_state = State.GEN_STRING
            else:
                self.buffer += f'"{p_name}": '
                self.current_state = State.GEN_NUMBER if \
                        p_prop.type == "number" else State.GEN_BOOLEAN
            return True

        if self.current_state == State.EMIT_PARAM_SEP:
            print(f"DEBUG: current_state={State.EMIT_PARAM_SEP}")
            print("DEBUG: Adding ',' to buffer...")
            self.buffer += ', '
            self.current_state = State.EMIT_PARAM_KEY
            return True

        if self.current_state == State.EMIT_END:
            self.buffer += '}}'
            self.current_state = State.END
            return True

        return False
    
    def get_allowed_token_ids(self) -> Set[int]:
        """Resolves all deterministic transitions and returns valid token IDs for 
            LLM states.
        """
        # TODO: Does it makes sense to keep the gen calls outside the loops?
        # Won't this actually be slower? Maybe merge both functions in one
        while self.advance_deterministic():
            pass
        print("DEBUG:\n--- INSIDE get_allowed_token_ids ---")
        allowed_ids: Set[int] = set()
        
        if self.current_state == State.SELECT_FUNCTION:
            fn_names = [fn_def.name for fn_def in self.functions]
            
            for fn_name in fn_names:
                if fn_name.startswith(self.fn_name_buffer):
                    # What we need to match next
                    target = fn_name[len(self.fn_name_buffer):]
                    
                    # Which vocab tokens are prefixes of target?
                    for token_id, token_str in self.vocab_mgr.id_to_token.items():
                        if target.startswith(token_str):
                            allowed_ids.add(token_id)
                
        elif self.current_state == State.GEN_STRING:
            # Does this really allow escaped quotes, \n, etc.??
            print(f"DEBUG: current_state: {self.current_state}")
            for token_id, token_str in self.vocab_mgr.id_to_token.items():
                clean = token_str.strip()
                if not clean:
                    continue
                if token_str == '"':
                    allowed_ids.add(token_id)
                elif not any(c in token_str for c in ['"', '\n', '\r', '\x00']):
                    allowed_ids.add(token_id)
        
        elif self.current_state == State.GEN_NUMBER:
            print(f"DEBUG: current_state: {self.current_state}")
            for token_id, token_str in self.vocab_mgr.id_to_token.items():
                clean = token_str.strip()
                if not clean:
                    continue
                # numeric token (digits or decimal)
                if clean.replace('.', '', 1).isdigit():
                    allowed_ids.add(token_id)
                # allow comma/brace only if we've already emitted digits
                elif clean in [',', '}'] and self._param_has_content:
                    allowed_ids.add(token_id)
        
        elif self.current_state == State.GEN_BOOLEAN:
            print(f"DEBUG: current_state: {self.current_state}")
            for token_id, token_str in self.vocab_mgr.id_to_token.items():
                clean = token_str.strip()
                if not clean:
                    continue
                if clean in ['true', 'false']:
                    allowed_ids.add(token_id)
                elif clean in [',', '}'] and self._param_has_content:
                    allowed_ids.add(token_id)

        return allowed_ids
    def update(self, token_id: int) -> None:
        """Appends chosen token to appropriate buffer and commits on completion.
        """
        token_str = self.vocab_mgr.id_to_token[token_id]

        is_param_state = self.current_state in (
            State.GEN_STRING,
            State.GEN_NUMBER,
            State.GEN_BOOLEAN
        )

        if is_param_state:
            self.param_value_buffer += token_str
        else:
            self.buffer += token_str

        if self.current_state == State.SELECT_FUNCTION:
            self.fn_name_buffer += token_str
            print(
                f"DEBUG update: fn_name_buffer='{self.fn_name_buffer}',"
                f" token_str='{token_str}'"
            )

            matching_fn = next(
                (f for f in self.functions if f.name == self.fn_name_buffer),
                None
            )
            print(f"DEBUG update: matching_fn={matching_fn}")
            if matching_fn:
                self.buffer += '"'
                self.selected_function = matching_fn
                self.parameter_queue = deque(
                    self.selected_function.parameters.items()
                )
                self.current_state = State.EMIT_PARAMS_HEADER
            elif not any(f.name.startswith(self.fn_name_buffer) \
                    for f in self.functions):
                raise CallMeError(
                    f"Invalid function name buffer: '{self.fn_name_buffer}'"
                )

        elif self.current_state == State.GEN_STRING and '"' in token_str:
            self._commit_param_value()
            self.buffer += '"'
            self.current_state = State.EMIT_PARAM_SEP \
                if self.parameter_queue else State.EMIT_END

        elif self.current_state in (State.GEN_NUMBER, State.GEN_BOOLEAN):
            if not (',' in token_str or '}' in token_str):
                self._param_has_content = True

            if ',' in token_str or '}' in token_str:
                # Strip trailing delimiter token before commit
                if self.param_value_buffer and self.param_value_buffer[-1] \
                        in (',', '}'):
                    self.param_value_buffer = self.param_value_buffer[:-1]

                self._commit_param_value()
                self.current_state = State.EMIT_PARAM_SEP if \
                    self.parameter_queue else State.EMIT_END

        while self.advance_deterministic():
            pass
   
    def get_current_state(self) -> State:
        """Compatibility helper used by tests: return current state enum."""
        return self.current_state
        # Is this really necessary?

    def is_complete(self) -> bool:
          """Check if the state machine has reached the END state.

          Returns:
              True if state machine is in END state, False otherwise.
          """
          return self.current_state == State.END
