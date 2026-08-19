from typing import Optional, List, Dict, Any, Set, Union
from enum import Enum, auto
import json
import re

from .schemas import FunctionDefinition  # ver se o flake8 deixa isso assim
from .vocabulary import VocabularyManager  # achei organizado
from .errors import CallMeError


class CurrentState(Enum):
    # FIXME: Rename this class?
    """Enumeration of possible states in the JSON state machine."""
    START = auto()
    CHOOSE_FUNCTION = auto()
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()
    END = auto()


class JSONStateMachine:
    """Tracks current state during token generation and determines allowed
            next tokens.
    """
    def __init__(
            self,
            prompt_txt: str,
            funcs: List[FunctionDefinition],
            vocab_mgr: VocabularyManager
        )-> None:
        """Initialize the state machine with prompt, functions, and vocabulary.

        Args:
            prompt_txt: The natural language user prompt.
            funcs: List of available function definitions.
            vocab_mgr: VocabularyManager instance for token lookups.
        """
        self.prompt_txt = prompt_txt
        self.functions = funcs
        self.vocab_mgr = vocab_mgr
        self.selected_function: Optional[FunctionDefinition] = None
        self.current_buffer: str = ""
        self.current_state: CurrentState = CurrentState.START
        self.param_queue: List[str] = []

    def get_current_state(self) -> CurrentState:
        """Return the current state of the state machine.

        Returns:
            The active CurrentState enum value.
        """
        return self.current_state

    def is_complete(self) -> bool:
        """Check if the state machine has reached the END state.

        Returns:
            True if state machine is in END state, False otherwise.
        """
        return self.current_state == CurrentState.END

    def update(self, token_input: Union[str, int]) -> None:
        """Update the state machine buffer with a newly generated token.

        Args:
            token_input: Token string representation or integer token ID.
        """
        if isinstance(token_input, int):
            token_str = self._get_token_str(token_input)
        else:
            token_str = token_input
        self.current_buffer += token_str
        self._update_internal_state()

    def _get_token_str(self, token_id: int) -> str:
        """Retrieve string representation for a given token ID.

        Args:
            token_id: Integer token ID.

        Returns:
            String token representation.
        """
        return self.vocab_mgr.id_to_token.get(token_id, "")

    def _get_vocab_id_dict(self) -> Dict[int, str]:
        """Retrieve the vocabulary mapping dictionary from vocabulary manager.

        Returns:
            Dictionary mapping token IDs to string tokens.
        """
        return getattr(self.vocab_mgr, "id_to_token", {}) # can I do that?

    def _update_internal_state(self) -> None:
        """Analyze current_buffer and update internal state and param queue."""

        if self.selected_function is None:
            for fn in self.functions:
                prefix = f'{{"name": "{fn.name}"'
                if prefix in self.current_buffer:
                    self.selected_function = fn
                    self.param_queue = list(fn.parameters.keys())
                    self.current_state = CurrentState.CHOOSE_FUNCTION
                    return # should really return not break?

        if self.selected_function is not None:
            if self.param_queue:
                current_pname = self.param_queue[0]
                p_type = self.selected_function.parameters[current_pname].type

                # FIXME: should also accept integer, float or some other names?
                # Maybe ignore case or something like that? Make it more robust

                if p_type == "number":
                    self.current_state = CurrentState.NUMBER
                if p_type == "string":
                    self.current_state = CurrentState.STRING
                if p_type == "boolean":
                    self.current_state = CurrentState.BOOLEAN

                if self._is_param_complete(current_pname, p_type):
                    self.param_queue.pop(0)  # FIXME: maybe use popletf?

        if not self.param_queue:
            if self.current_buffer.rstrip().endswith("}}"):
                self.current_state = CurrentState.END

    def _is_param_complete(self, p_name: str, p_type: str) -> bool:
        """Check if parameter value has been fully generated in current_buffer.

        Args:
            pname: Parameter key name.
            ptype: Expected parameter type ("number", "string", "boolean").

        Returns:
            True if parameter value is complete, False otherwise.
        """
        if p_type == "boolean":
            patt = r'"' + re.escape(p_name) + r'":\s*(true|false)\s*(,|})'
        elif p_type == "number":
            patt = r'"' + re.escape(p_name) + r'":\s*(-?\d+(?:\.\d+)?)\s*(,|})'
        elif p_type == "string":
            patt = r'"' + re.escape(p_name) + r'":\s*"([^"\\]*)"\s*(,|})'
        else:
            return False
        return bool(re.search(patt, self.current_buffer))

    def get_allowed_token_ids(self) -> Set[int]:
        """Determine allowed next token IDs based on current state and buffer.

        Returns:
            Set of valid token IDs allowed for the next generation step.
        """
        vocab_dict = self._get_vocab_id_dict()
        if not vocab_dict:
            return set()
        if self.current_state == CurrentState.END:
            return set()
        allowed_ids: Set[int] = set()
        for token_id, token_str in vocab_dict.items():
            if self._is_candidate_valid(token_str):
                allowed_ids.add(token_id)
        return allowed_ids

    def _is_candidate_valid(self, candidate_str: str) -> bool:
        """Check if appending candidate_str to current_buffer produces a valid
            prefix.

        Args:
            candidate_str: The candidate token string.

        Returns:
            True if candidate_str maintains schema validity, False otherwise.
        """

        combined = self.current_buffer + candidate_str
        if self.selected_function is None:
            for fn in self.functions:
                target = f'{{"name": "{fn.name}", "parameters": {{'
                if self._matches_prefix(combined, target):
                    return True
            return False

        fn = self.selected_function
        header_target = f'{{"name": "{fn.name}", "parameters": {{'

        if not self.current_buffer.startswith(header_target):
            return self._matches_prefix(combined, header_target)
        if not self.param_queue:
            full_target = f"{header_target}}}"
            return self._matches_prefix(combined, full_target)

        current_pname = self.param_queue[0]
        p_type = fn.parameters[current_pname].type
        key_prefix = f'"{current_pname}": '

        if key_prefix not in self.current_buffer:
            return self._matches_prefix(combined, key_prefix)

        sep = ", " if len(self.param_queue) > 1 else "}}"

        if p_type == "boolean":
            val_targets = [f"{key_prefix}true{sep}", f"{key_prefix}false{sep}"]
            return any(self._matches_prefix(combined, t) for t in val_targets)
        if p_type == "number":
            val = combined[combined.rfind(key_prefix) + len(key_prefix) :]
            if not val:
                return True
            clean_val = val.rstrip(" ,}")
            if clean_val and not re.match(r"^-?\d*(\.\d*)?$", clean_val):
                return False
            return True
        if p_type == "string":
            val = combined[combined.rfind(key_prefix) + len(key_prefix) :]
            if not val:
                return True
            if not val.startswith('"'):
                return val.strip() == "" or val == '"'
            return True

        return True

    @staticmethod
    def _matches_prefix(combined_str: str, target_str: str) -> bool:
        """Helper to test prefix compatibility between combined string and target.

        Args:
            combined_str: Buffer + candidate token string.
            target_str: Expected target string template.

        Returns:
            True if combined_str is compatible with target_str.
        """
        if target_str.startswith(combined_str) or combined_str.startswith(
                target_str
                ):
            return True

        norm_combined = "".join(combined_str.split())
        norm_target = "".join(target_str.split())
        return norm_target.startswith(norm_combined) or norm_combined.startswith(
            norm_target
        )

    def get_result_dict(self) -> Dict[str, Any]:
        """Parse the generated buffer into a valid JSON object dictionary.

        Returns:
            Dictionary containing 'name' and 'parameters'.

        Raises:
            CallMeError: If current_buffer cannot be parsed as valid JSON.
        """
        try:
            data = json.loads(self.current_buffer)
            if isinstance(data, dict):
                return data
            # FIXME: replace CallMeError with more specific custom error type
            raise CallMeError("Output buffer is not a valid JSON object.")
        except json.JSONDecodeError as e:
            raise CallMeError(f"Failed to parse generated JSON buffer: {e}")

