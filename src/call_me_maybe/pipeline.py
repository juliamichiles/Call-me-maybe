from typing import List, Set, Dict, Any, TYPE_CHECKING
import numpy as np 

from .schemas import FunctionDefinition 
from .state_machine import JSONStateMachine
from .vocabulary import VocabularyManager
from .errors import CallMeError
from .state_machine import JSONStateMachine

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def select_next_token(logits: List[float], allowed_ids: Set[int]) -> int:
    """Masks logits for invalid tokens and returns the token ID with the
            highest score.
    """
    logits_arr = np.array(logits)
    constrained_logits = np.full_like(logits_arr, -np.inf)

    valid_indices = list(allowed_ids)
    if not valid_indices:
        # TODO: replace with custom error?
        raise RuntimeError(
                "No valid tokens avaliable to satisfy schema"
        )
    constrained_logits[valid_indices] = logits_arr[valid_indices]
    return int(np.argmax(constrained_logits))


class Generation:
    @staticmethod
    def _format_prompt(
            user_prompt: str, 
            functions: List[FunctionDefinition]
            ) -> str:
        funcs_json = [f.model_dump() for f in functions]
    
        formatted = (
                "You are a helpful assistant that outputs function calls in "
                "strict JSON format.\n"
                f"Available Functions:\n{funcs_json}\n\n"
                f"User Prompt: {user_prompt}\n\n"
                "Output JSON:"
        )
        return formatted

    def gen_function_call(
            self,
            model: "Small_LLM_Model",  # Keep as a str even with TYPE_CHECKING?
                                       # Shouldn't instantiate actual model???
            prompt_txt: str,
            functions: List[FunctionDefinition],
            vocab_mgr: VocabularyManager,
            max_tokens: int = 150
    ) -> Dict[str, Any]:
        """Generates a schema-compliant JSON function call for a single 
                prompt.
        """
        formatted_prompt = self._format_prompt(prompt_txt, functions)
        input_ids: List[int] = model.encode(formatted_prompt).tolist()[0]
        state_machine = JSONStateMachine(prompt_txt, functions, vocab_mgr)

        for _ in range(max_tokens):
            if state_machine.is_complete():
                break
            logits = model.get_logits_from_input_ids(input_ids)
            allowed_ids = state_machine.get_allowed_token_ids()
            # debug: inspect why allowed_ids might be empty
            print("DEBUG: buffer:", repr(state_machine.current_buffer))
            print("DEBUG: state:", state_machine.current_state)
            print("DEBUG: selected_function:", getattr(state_machine.selected_function, "name", None))
            print("DEBUG: candidate_allowed_count:", len(allowed_ids))
            if len(allowed_ids) > 0:
                # show a small sample of token strings for context
                sample = list(allowed_ids)[:20]
                print("DEBUG: allowed sample (id, str):", [(i, vocab_mgr.id_to_token[i]) for i in sample])
            else:
                # also show candidates from _get_candidate_tokens() to see why filtering removed them
                candidates = state_machine._get_candidate_tokens()
                print("DEBUG: candidate_count:", len(candidates))
                print("DEBUG: candidate_sample (id, str):", [(i, vocab_mgr.id_to_token[i]) for i in list(candidates)[:20]])
                # show top-scoring logits for candidate ids so we know model scores (needs logits arr)
                import numpy as _np
                logits_arr = _np.array(logits)
                cand_scores = [(int(i), float(logits_arr[i])) for i in list(candidates)[:50] if i < len(logits_arr)]
                print("DEBUG: candidate_scores_sample:", cand_scores)
            next_token = select_next_token(logits, allowed_ids)
            state_machine.update(next_token)
            input_ids.append(next_token)

        result = state_machine.get_result_dict()
        print(result)
        return {
                "prompt": prompt_txt,
                "name": result.get("name", ""),
                "parameters": result.get("parameters", {})
        }
