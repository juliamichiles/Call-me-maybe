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
            next_token = select_next_token(logits, allowed_ids)
            state_machine.update(next_token)
            input_ids.append(next_token)

        result = state_machine.get_result_dict()
        return {
                "prompt": prompt_txt,
                "name": result.get("name", ""),
                "parameters": result.get("parameters", {})
        }
