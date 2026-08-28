from functools import lru_cache
from typing import Optional, List, Dict, Any, Set, Union
from enum import Enum, auto
import json
import re

from .schemas import FunctionDefinition
from .vocabulary import VocabularyManager
from .errors import CallMeError

# Compiled module-level regexes to avoid repeated compilation
NUM_RE = re.compile(r'"(?P<name>[^"\\]*)":\s*(-?\d+(?:\.\d+)?)\s*(,|})')
BOOL_RE = re.compile(r'"(?P<name>[^"\\]*)":\s*(true|false)\s*(,|})', re.IGNORECASE)
STRING_RE = re.compile(r'"(?P<name>[^"\\]*)":\s*"([^"\\]*)"\s*(,|})')
# Fallback small numeric pattern used in incremental checks
INCR_NUM_RE = re.compile(r"^-?\d*(\.\d*)?$")

# FIXME: Regex patterns might be too naif, don't handle escaped chars properly
# FIXME: I think it should handle more words than number, boolean and string

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
        self.value_tokens: Set[int] = set()
        self._precompute_value_tokens()

        # per-instance cache: remainder -> tuple(ids) (LRU)
        # this avoids global/class-level cache and the need to clear it externally
        self._cached_candidates_for_remainder = lru_cache(maxsize=4096)(
            lambda rem: tuple(self._compute_candidates_for_remainder(rem))
        )

        # Precompute a smaller token subset that's reasonable inside JSON strings.
        # This avoids returning the whole vocabulary for STRING state.
        self.string_tokens: Set[int] = set()
        for tid, tstr in self.vocab_mgr.id_to_token.items():
            # Accept tokens that include quotes, escapes, whitespace, alphabetic content,
            # or are short punctuation that commonly appears inside strings.
            s = tstr
            if (
                '"' in s or "\\" in s
                or s.strip() == ""
                or any(ch.isalpha() for ch in s)
                or len(s) <= 2 and any(ch in s for ch in ("-", "_", ".", "'"))
            ):
                self.string_tokens.add(tid)

        # memoized validator keyed on (short buffer tail, candidate)
        # we use a small LRU to reduce repeated expensive validation work
        self._is_candidate_valid_cached = lru_cache(maxsize=8192)(
            lambda buf_tail, cand: self._is_candidate_valid_uncached(buf_tail, cand)
        )

    def _precompute_value_tokens(self) -> None:
        """Build a coarse candidate set once to avoid scanning the vocabulary
            on every generation step.
        """
        # matches digits, ., -, spaces, commas, braces
        numeric_re = re.compile(r'^[\d\.\-\s,}]+$')
        boolean_re = re.compile(r'^(?:true|false)$', re.IGNORECASE)
        for token_id, token_str in self.vocab_mgr.id_to_token.items():
            if numeric_re.match(token_str) or boolean_re.match(token_str):
                self.value_tokens.add(token_id)

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

    def _update_internal_state(self) -> None:
        """Analyze current_buffer and update internal state and param queue."""

        if self.selected_function is None:
            for fn in self.functions:
                prefix = f'{{"name": "{fn.name}"'
                if prefix in self.current_buffer:
                    self.selected_function = fn
                    self.param_queue = list(fn.parameters.keys())
                    self.current_state = CurrentState.CHOOSE_FUNCTION
                    break  # should really break not return?

        if self.selected_function is not None:
            if self.param_queue:
                current_pname = self.param_queue[0]
                p_type = self.selected_function.parameters[current_pname].type

                # FIXME: should also accept integer, float or some other names?
                # Maybe ignore case or something like that? Make it more robust

                if p_type == "number":
                    self.current_state = CurrentState.NUMBER
                elif p_type == "string":
                    self.current_state = CurrentState.STRING
                elif p_type == "boolean":
                    self.current_state = CurrentState.BOOLEAN

                if self._is_param_complete(current_pname, p_type):
                    self.param_queue.pop(0)  # FIXME: maybe use popletf?

        if not self.param_queue:
            if self.current_buffer.rstrip().endswith("}}"):
                self.current_state = CurrentState.END

    def _compute_candidates_for_remainder(self, remainder: str) -> Set[int]:
        """Return a set of token ids matching the remainder using Trie and token map.

        This is the heavy operation we cache per-instance.
        """
        ids: Set[int] = set()
        # tokens_for_prefix should ideally be a cached per-instance wrapper in VocabularyManager
        ids.update(self.vocab_mgr.tokens_for_prefix(remainder))
        # check shorter prefix subs (as before) but avoid excessive allocations
        rl = len(remainder)
        for i in range(1, rl + 1):
            prefix_sub = remainder[:i]
            token_set = self.vocab_mgr.token_to_id.get(prefix_sub)
            if token_set:
                ids.update(token_set)
        return ids

    def _get_candidate_tokens(self) -> Set[int]:
        """Dramatically reduces the search space using the VocabularyTrie."""
        # STRING: return a limited set instead of the entire vocabulary
        if self.current_state == CurrentState.STRING:
            return set(self.string_tokens)

        candidate_ids: Set[int] = set()

        if self.current_state in (CurrentState.NUMBER, CurrentState.BOOLEAN):
            candidate_ids.update(self.value_tokens)

        targets: List[str] = []
        if self.selected_function is None:
            for fn in self.functions:
                targets.append(f'{{"name": "{fn.name}", "parameters": {{')
        else:
            fn = self.selected_function
            header = f'{{"name": "{fn.name}", "parameters": {{'
            if not self.current_buffer.startswith(header):
                targets.append(header)
            elif not self.param_queue:
                targets.append('}}')
            else:
                current_pname = self.param_queue[0]
                targets.append(f'"{current_pname}": ')
                targets.append(', "')
                targets.append('}}')

        # limit buffer suffix length to reduce repeated large allocations
        buf_tail = self.current_buffer[-128:]

        for target in targets:
            overlap_len = 0
            max_check = min(len(buf_tail), len(target))
            # Find the longest suffix of the buffer that matches a prefix of the target
            for i in range(1, max_check + 1):
                if target.startswith(buf_tail[-i:]):
                    overlap_len = i
            remainder = target[overlap_len:]
            if remainder:
                # use the per-instance cached helper
                candidate_ids.update(self._cached_candidates_for_remainder(remainder))

        return candidate_ids

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
        """Determine allowed next token IDs using Trie to optimize Token selection.

        Returns:
            Set of valid token IDs allowed for the next generation step.
        """
        if self.current_state == CurrentState.END:
            return set()
        candidate_ids = self._get_candidate_tokens()
        allowed_ids: Set[int] = set()
       
        # if NUM or BOOL, allow any candidate in the precomputed value_tokens
        if self.current_state in (CurrentState.NUMBER, CurrentState.BOOLEAN):
            numeric_candidates = candidate_ids & self.value_tokens
            if numeric_candidates:
                return numeric_candidates
            # fallback in case no num candidates are found
            candidate_ids = set(list(candidate_ids)[:2000])
 
        # when all params generated, allow structural closing tokens (}, }, comma, whitespace)
        if not self.param_queue:
            structural_allowed = set()
            for tid in candidate_ids:
                s = self.vocab_mgr.id_to_token[tid].strip()
                if s in ("}", "},", ",", "") or s.startswith("}"):
                    structural_allowed.add(tid)
                if s in ('"', ' ', "'"):
                    structural_allowed.add(tid)
            if structural_allowed:
                allowed_ids.update(structural_allowed)
                if allowed_ids == candidate_ids:
                    return allowed_ids
        
        remaining_to_check = list(candidate_ids - allowed_ids)
        MAX_CHECK = 2000
        if len(remaining_to_check) > MAX_CHECK:
            remaining_to_check = remaining_to_check[:MAX_CHECK]

        for token_id in remaining_to_check:
            token_str = self.vocab_mgr.id_to_token[token_id]
            # use cached validator which keys on a short buffer tail + candidate
            if self._is_candidate_valid(token_str):
                allowed_ids.add(token_id)
                # keep debugging prints optional; remove or gate them later
                print(f"current token: [{token_id}] {token_str} -", end=" ")
                print("[VALID]")
        
        # fallback in case allowed_ids is empty for numeric/boolean
        if not allowed_ids and self.current_state \
                in (CurrentState.NUMBER, CurrentState.BOOLEAN):
            return candidate_ids & self.value_tokens

        return allowed_ids
    


    def _is_candidate_valid_uncached(self, buffer_tail: str, candidate_str: str) -> bool:
        """Uncached heavy validator. buffer_tail is a short suffix of current_buffer."""
        combined = buffer_tail + candidate_str

        # If no function selected, only allow header prefixes that match any function
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

        # If key not yet present in buffer, check if candidate helps form the key
        if key_prefix not in self.current_buffer:
            overlap_len = 0
            max_check = min(len(self.current_buffer), len(key_prefix))
            for i in range(1, max_check + 1):
                if key_prefix.startswith(self.current_buffer[-i:]):
                    overlap_len = i
            remainder = key_prefix[overlap_len:]
            cand_norm = "".join(candidate_str.split())
            rem_norm = "".join(remainder.split())
            if rem_norm.startswith(cand_norm) or cand_norm.startswith(rem_norm[: len(cand_norm)]):
                return True
            return self._matches_prefix(combined, key_prefix)

        sep = ", " if len(self.param_queue) > 1 else "}}"

        # boolean: allow prefixes of true/false
        if p_type == "boolean":
            val_targets = [f"{key_prefix}true{sep}", f"{key_prefix}false{sep}"]
            return any(self._matches_prefix(combined, t) for t in val_targets)

        # number: incremental numeric validation
        if p_type == "number":
            val = combined[combined.rfind(key_prefix) + len(key_prefix) :]
            if not val:
                return True
            clean_val = val.rstrip(" ,}")
            if clean_val and not INCR_NUM_RE.match(clean_val):
                return False
            return True

        # string: basic checks (imperfect, but fast)
        if p_type == "string":
            val = combined[combined.rfind(key_prefix) + len(key_prefix) :]
            if not val:
                return True
            if not val.startswith('"'):
                return val.strip() == "" or val == '"'
            return True

        return True

    def _is_candidate_valid(self, candidate_str: str) -> bool:
        """Cached wrapper for candidate validation. Uses a short buffer tail as cache key."""
        tail = self.current_buffer[-128:]
        return self._is_candidate_valid_cached(tail, candidate_str)

    @staticmethod
    def _matches_prefix(combined_str: str, target_str: str) -> bool:
        """Helper to test prefix compatibility between combined string and target.

        Args:
            combined_str: Buffer + candidate token string.
            target_str: Expected target string template.

        Returns:
            True if combined_str is compatible with target_str.
        """
        if target_str.startswith(combined_str) or combined_str.startswith(target_str):
            return True

        norm_combined = "".join(combined_str.split())
        norm_target = "".join(target_str.split())
        return norm_target.startswith(norm_combined) or norm_combined.startswith(norm_target)

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
            snippet = self.current_buffer[:1000]  # truncate if extremely long
            raise CallMeError(
                    f"Failed to parse generated JSON buffer: {e}"
                    f"\nBuffer snippet: {snippet!r}"
            )
