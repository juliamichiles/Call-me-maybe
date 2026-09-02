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
    def __init__(self, functions: List[FunctionDefinition]) -> None:
        self.functions = functions

    def _format_prompt(self, user_input: str) -> str:
        system_guide = (
                "You are a strictly-compliant JSON assistant.\n"
                "You must answer the user's request by calling exactly one"
                " function from the list below. Output must be valid JSON with"
                " keys: \"name\" and \"parameters\".\n\n"
                "IMPORTANT: If the user's prompt contains numbers or quoted"
                " strings, you MUST copy those exact values into the parameters"
                "Do not invent numbers.\n"
                "If a string contains quotes, escape them"
                " (e.g. \\\"hello\\\").\n\n"
                "Examples (Input -> Assistant JSON):\n"
                "User Request: What is the square root of 16?\n"
                "Assistant Response: "
                "{\"name\": \"fn_get_square_root\", \"parameters\": "
                "{\"a\": 16}}\n\n"
                "User Request: What is the sum of 2 and 3?\n"
                "Assistant Response: "
                "{\"name\": \"fn_add_numbers\", \"parameters\": "
                "{\"a\": 2, \"b\": 3}}\n\n"
                "User Request: Greet \"John\" please\n"
                "Assistant Response: " 
                "{\"name\": \"fn_greet\", \"parameters\": {\"name\":"
                " \"John\"}}\n\n"
                "Available functions:\n"
        )
        function_lines = []
        
        for fn in self.functions:
            params = []
            for p_name, p_def in fn.parameters.items():
                params.append(f"{p_name}: {p_def.type}")
            params = ", ".join(params)
            function_lines.append(f"- {fn.name}({params}): {fn.description}")
        functions_str = "\n".join(function_lines)
        
        full_prompt = (
                f"{system_guide}{functions_str}\n\n"
                f"User Request: {user_input}\nAssistant Response:\n{{"
        )
        return full_prompt

    def gen_function_call(
            self,
            model: "Small_LLM_Model",
            prompt_txt: str,
            vocab_mgr: VocabularyManager,
            max_tokens: int = 150
            ) -> Dict[str, Any]:
        """Generates a structured JSON function call using token-by-token 
                constrained decoding.
        """

        formated_prompt = self._format_prompt(prompt_txt)
        input_ids: List[int] = model.encode(formated_prompt).tolist()[0]
        state_machine = JSONStateMachine(self.functions, vocab_mgr)
        
        print(f"DEBUG: max_tokens: {max_tokens}")
        for _ in range(max_tokens):
            allowed_ids = state_machine.get_allowed_token_ids()
            print(f"DEBUG: allowed_ids: {allowed_ids}") 
            if state_machine.is_complete():
                print("DEBUG: state:", state_machine.current_state)
                print(f"DEBUG: state machine is complete: {state_machine.is_complete()}")
                break

            logits = model.get_logits_from_input_ids(input_ids)
            print(f"DEBUG: logitos: {logits[0]} -> {logits[-1]}")
            next_token = select_next_token(logits, allowed_ids)
            print(f"DEBUG: next token: {next_token}")
            # debug: inspect why allowed_ids might be empty
            print("DEBUG: buffer:", repr(state_machine.buffer))
            print("DEBUG: state:", state_machine.current_state)
            print("DEBUG: selected_function:", getattr(
               state_machine.selected_function, "name", None
            ))
            print("DEBUG: candidate_allowed_count:", len(allowed_ids))
            state_machine.update(next_token)
            input_ids.append(next_token)

            print(f"DEBUG: sm_buffer={state_machine.buffer}")
        try:
            parsed_output = json.loads(state_machine.buffer)
            return {
                "prompt": prompt_txt,
                "name": parsed_output.get("name", ""),
                "parameters": parsed_output.get("parameters", {})
            }
        except json.JSONDecodeError as e:
            raise CallMeError(f"Generated output failed JSON parsing: {e}")
