import json
from typing import List, Dict, Any, Set, TYPE_CHECKING
import numpy as np 

from .schemas import FunctionDefinition
from .state_machine import JSONStateMachine
from .vocabulary import VocabularyManager
from .errors import CallMeError
from src.call_me_maybe import state_machine

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def select_next_token(logits: List[float], allowed_ids: Set[int]) -> int:
    """Masks unallowed logits with negative infinity and returns the top token ID.
    """
    if not allowed_ids:
        raise CallMeError(
                "No valid tokens avaliable to satisfy current schema state"
        )
    
    logits_arr = np.array(logits, dtype=np.float32)
    contrained_logits = np.full_like(logits_arr, -np.inf)

    valid_indices = list(allowed_ids)
    contrained_logits[valid_indices] = logits_arr[valid_indices]
    return int(np.argmax(contrained_logits))


class Generation:
    def gen_function_call(
            self,
            model: "Small_LLM_Model",
            prompt_txt: str,
            functions: List[FunctionDefinition],
            vocab_mgr: VocabularyManager,
            max_tokens: int = 150
            ) -> Dict[str, Any]:
        """Generates a structured JSON function call using token-by-token 
                constrained decoding.
        """

        formated_prompt = f"User: {prompt_txt}\nOutput:"
        input_ids: List[int] = model.encode(formated_prompt).tolist()[0]
        state_machine = JSONStateMachine(prompt_txt, functions, vocab_mgr)

        for _ in range(max_tokens):
            allowed_ids = state_machine.get_allowed_token_ids()
            if state_machine.is_complete():
                break

            logits = model.get_logits_from_input_ids(input_ids)
            next_token = select_next_token(logits, allowed_ids)

            state_machine.update(next_token)
            input_ids.append(next_token)

        try:
            parsed_output = json.loads(state_machine.buffer)
            return {
                "prompt": prompt_txt,
                "name": parsed_output.get("name", ""),
                "parameters": parsed_output.get("parameters", {})
            }
        except json.JSONDecodeError as e:
            raise CallMeError(f"Generated output failed JSON parsing: {e}")
