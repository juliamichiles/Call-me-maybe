from typing import List, Set
import numpy as np 

from .schemas import FunctionDefinition 
from .state_machine import JSONStateMachine


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


# TODO: Either add more stuff here or just keep it as a function
class Generation:
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
