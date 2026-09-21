from enum import Enum, auto
import enum
# from os import remove
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Set, Optional
from collections import deque
# import time
# from typing_extensions import TypeVarTuple

from .vocabulary import VocabularyManager
from .schemas import FunctionDefinition, ParameterProperty
from .errors import CallMeError

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


class State(Enum):
    """ Enumerations for each possible generation state to further determine if
            LLM should be called or bypassed in a given state.
    """
    # --- FORCED / DETERMINISTIC STATES ---
    EMIT_START = auto()
    EMIT_PARAMS_HEADER = auto()
    EMIT_PARAM_KEY = auto()
    EMIT_PARAM_SEP = auto()
    EMIT_END = auto()

    # --- LLM-DRIVEN STATES ---
    SELECT_FUNCTION = auto()
    SELECT_PARAMETER_VALUE = auto()

    # --- TERMINAL STATE ---
    END = auto() 

class JSONStateMachine:
    """Tracks current state during token generation and determines allowed
              next tokens.
      """
    def __init__(
            self,
            prompt_txt: str,
            functions: List[FunctionDefinition],
            model: "Small_LLM_Model",
            vocab_mgr: VocabularyManager
    )-> None:
        """Initialize the state machine with prompt, functions, and vocabulary.

          Args:
              prompt_txt: The natural language user prompt.
              functions: List of available function definitions.
              model: The LLM model instance.
              vocab_mgr: VocabularyManager instance for token lookups.
          """
        self.prompt_txt = prompt_txt
        self.functions = functions
        self.vocab_mgr = vocab_mgr
        self.model = model 
        self.deterministic_token_ids: List[int] = []
        self.current_state = State.EMIT_START
        self.selected_function: Optional[FunctionDefinition] = None
        self.parameter_queue: deque[Tuple[str, ParameterProperty]] = deque()
        self._param_has_content = False
        self._escape_active = False
        self._string_open = False
        self.fn_name_buffer = ""
        self.param_value_buffer = ""
        self.current_param_name = ""  # Track which parameter we're filling
        self.current_param_type = ""  # Track the type of current parameter
        self.buffer = ""

    def _encoded(self, text: str) -> List[int]:
       return self.model.encode(text).tolist()[0]
     
    def _commit_param_value(self) -> None:
        """Appends accumulated parameter value to main buffer and resets local
                buffer.
        """
        self.buffer += self.param_value_buffer
        self.param_value_buffer = ""

    def advance_deterministic(self) -> Optional[List[int]]:
        """Bypasses LLM by appending mandatory syntax tokens directly.
                Returns the encoded token IDs of text appended,
                or None if no state change.
        """
        appended_text = ""

        if self.current_state == State.EMIT_START:
            appended_text = '{"name": "'
            self.buffer += appended_text
            self.current_state = State.SELECT_FUNCTION
            return self._encoded(appended_text)

        if self.current_state == State.EMIT_PARAMS_HEADER:
            appended_text = '", "parameters": {'
            self.buffer += appended_text
            self.current_state = State.EMIT_PARAM_KEY if self.parameter_queue \
                    else State.END
            return self._encoded(appended_text)

        if self.current_state == State.EMIT_PARAM_KEY:
            p_name, p_prop = self.parameter_queue.popleft()
            self._param_has_content = False
            self._string_open = False
            self._escape_active = False
            self.param_value_buffer = ""
            self.current_param_name = p_name
            self.current_param_type = p_prop.type
            
            appended_text = f'"{p_name}": '
            self.buffer += appended_text
            
            # All parameter values now go through SELECT_PARAMETER_VALUE
            # regardless of type. The type constraint is applied in 
            # get_allowed_token_ids()
            self.current_state = State.SELECT_PARAMETER_VALUE
            return self._encoded(appended_text)
            
        if self.current_state == State.EMIT_PARAM_SEP:
            appended_text = ', '
            self.buffer += appended_text
            self.current_state = State.EMIT_PARAM_KEY
            return self._encoded(appended_text)

        if self.current_state == State.EMIT_END:
            appended_text = '}}'
            self.buffer += appended_text
            self.current_state = State.END
            return self._encoded(appended_text)
            
        return None
        
    def _get_generation_context(self) -> Dict[str, Any]:
        """Build context dict for the current generation state.
            Returns info about:
                - The original prompt
                - What's been generated so far
                - Current parameter being filled
        """
        # UNUSED!!
        return {
                "prompt_txt": self.prompt_txt,
                "current_buffer": self.buffer,
                "param_value_buffer": self.param_value_buffer,
                "current_param_name": self.current_param_name,
                "current_param_type": self.current_param_type,
                "deterministic_token_ids": self.deterministic_token_ids,
                "current_state": self.current_state,
                "selected_function": self.selected_function.name \
                        if self.selected_function else None
        }

    def get_allowed_token_ids(self) -> Set[int]:
        """Resolves all deterministic transitions and returns valid token IDs for 
            LLM states.
        """
        # t_start = time.perf_counter()
        while True:
            det_tokens = self.advance_deterministic()
            if det_tokens is None:
                break
            self.deterministic_token_ids.extend(det_tokens)

        # print("DEBUG:\n--- INSIDE get_allowed_token_ids ---")
        # print(f"DEBUG: current_state = {self.current_state}")
        allowed_ids: Set[int] = set()
        
        if self.current_state == State.SELECT_FUNCTION:
            # t_fn_start = time.perf_counter()
            allowed_ids = self._get_allowed_function_tokens()
            # t_fn_end = time.perf_counter()
            # print(
            #         "[TIMING.SM] _get_allowed_function_tokens: "
            #         f"{(t_fn_end - t_fn_start)*1000000:.2f}µs",
            #         file=sys.stderr
            # )
            
        elif self.current_state == State.SELECT_PARAMETER_VALUE:
            # t_param_start = time.perf_counter()
            allowed_ids = self._get_allowed_parameter_value_tokens()
            # t_param_end = time.perf_counter()
            # print(
            #         "[TIMING.SM] _get_allowed_parameter_value_tokens: "
            #         f"{(t_param_end - t_param_start)*1000000:.2f}µs", 
            #         file=sys.stderr
            # )
        # t_end = time.perf_counter()
        # print(
        #         "[TIMING.SM] get_allowed_token_ids total: "
        #         f"{(t_end - t_start)*1000000:.2f}µs", 
        #         file=sys.stderr
        # )
        return allowed_ids
    
    def _get_allowed_function_tokens(self) -> Set[int]:
        """Return tokens that can continue the current function name."""
        allowed_ids: Set[int] = set()
        
        for function in self.functions:
            name = function.name
            if not name.startswith(self.fn_name_buffer):
                continue
            remaining = name[len(self.fn_name_buffer):]
            allowed_ids.update(
                self.vocab_mgr.token_ids_that_prefix(remaining)
            )

        return allowed_ids

    def _get_allowed_parameter_value_tokens(self) -> Set[int]:
        """Return tokens valid for the current parameter value."""

        parameter_type = self.current_param_type.lower()

        if parameter_type == "string":
            return self._get_allowed_string_tokens()
        if parameter_type in ("number", "integer"):
            return self._get_allowed_number_tokens()
        if parameter_type == "boolean":
            return self._get_allowed_boolean_tokens()
        # FIXME: Is that right?? Or should I handle other types differently?
        raise CallMeError(
            f"Unsupported parameter type: {self.current_param_type}"
        )   
    
    def _get_allowed_string_tokens(self) -> Set[int]:
        """O(1) lookup using precomputed token sets."""
        if not self._string_open:
            return self.vocab_mgr.quote_ids
        
        return self.vocab_mgr.valid_string_all_ids
    
    def _get_allowed_number_tokens(self) -> Set[int]:
        """O(1) lookup using precomputed number sets."""
        if not self.param_value_buffer:
            return set(self.vocab_mgr.number_start_ids)

        allowed_ids = set(self.vocab_mgr.number_body_ids)
        if self._param_has_content:
            allowed_ids.update(self.vocab_mgr.delimiter_ids)
        return allowed_ids

    def _get_allowed_boolean_tokens(self) -> Set[int]:
        """Return tokens that can continue true/false."""

        if not self.param_value_buffer:
            allowed_ids: Set[int] = set()
            allowed_ids.update(
                self.vocab_mgr.token_ids_that_prefix("true")
            )
            allowed_ids.update(
                self.vocab_mgr.token_ids_that_prefix("false")
            )
            return allowed_ids

        allowed_ids = set()

        for value in ("true", "false"):
            if value.startswith(self.param_value_buffer):
                remaining = value[len(self.param_value_buffer):]
                allowed_ids.update(
                    self.vocab_mgr.token_ids_that_prefix(remaining)
                )
        if self._param_has_content:
            allowed_ids.update(self.vocab_mgr.delimiter_ids)

        return allowed_ids
    
    def update(self, token_id: int) -> None:
        """Appends chosen token to appropriate buffer and commits on completion.
        """
        token_str = self.vocab_mgr.id_to_token[token_id]

        # All parameter value states go into param_value_buffer
        if self.current_state == State.SELECT_PARAMETER_VALUE:
            self.param_value_buffer += token_str
        else:
            self.buffer += token_str

        if self.current_state == State.SELECT_FUNCTION:
            self._handle_function_selection(token_str)
            
        elif self.current_state == State.SELECT_PARAMETER_VALUE:
            self._handle_parameter_value_selection(token_str)

        # Try to advance deterministically
        while True:
            det_tokens = self.advance_deterministic()
            if det_tokens is None:
                break
            self.deterministic_token_ids.extend(det_tokens)

    def _handle_function_selection(self, token_str: str) -> None:
        """Process token during function name selection."""
        self.fn_name_buffer += token_str
        # print(
        #    f"DEBUG update: fn_name_buffer='{self.fn_name_buffer}',"
        #    f" token_str='{token_str}'"
        # )

        matching_fn = next(
            (f for f in self.functions if f.name == self.fn_name_buffer),
            None
        )
        # print(f"DEBUG update: matching_fn={matching_fn}")
        if matching_fn:
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

    def _handle_parameter_value_selection(self, token_str: str) -> None:
        """Process token during parameter value selection.
        """
        if self.current_param_type == "string":
            if not self._string_open:
                if '"' in token_str:
                    self._string_open = True
                    idx = token_str.find('"')
                    token_str = token_str[idx + 1:]
                else:
                    return
            if self._string_open:
                for i, char in enumerate(token_str):
                    if self._escape_active:
                        self._escape_active = False
                    elif char == '\\':
                        self._escape_active = True
                    elif char == '"':
                        # closing quote, truncate trailing chars
                        rm_len = len(token_str) - i - 1
                        if rm_len > 0:
                            self.param_value_buffer = \
                                    self.param_value_buffer[:-rm_len]
                        self._commit_param_value()
                        self._string_open = False
                        self._param_has_content = False
                        self._escape_active = False
                        self.current_state = (
                                State.EMIT_PARAM_SEP
                                if self.parameter_queue
                                else State.EMIT_END
                        )
                        return
                self._param_has_content = True
        else:
            if self.current_param_type in ("number", "boolean"):
                if token_str.strip() and token_str.strip() not in [',', '}']:
                    self._param_has_content = True
                if token_str.strip() in [',', '}'] and self._param_has_content:
                    if self.param_value_buffer.endswith(token_str):
                        self.param_value_buffer = self.param_value_buffer[:-len(token_str)]

                    self._commit_param_value()
                    self._param_has_content = False
                    self.current_state = State.EMIT_PARAM_SEP \
                        if self.parameter_queue else State.EMIT_END

    def get_current_state(self) -> State:
        """Compatibility helper used by tests: return current state enum."""
        return self.current_state

    def is_complete(self) -> bool:
        """Check if the state machine has reached the END state.

        Returns:
            True if state machine is in END state, False otherwise.
        """
        return self.current_state == State.END

    @property
    def get_full_buffer(self) -> str:
        """Returns the current accumulated JSON buffer string."""
        return self.buffer + self.param_value_buffer
