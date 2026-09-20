import json
import time
from typing import List, Dict, Any, Set, TYPE_CHECKING
import numpy as np


from .schemas import FunctionDefinition
from .state_machine import JSONStateMachine
from .vocabulary import VocabularyManager
from .errors import CallMeError
from .visualization import GenVisualizer
# from src.call_me_maybe import state_machine

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
    def __init__(
            self, 
            functions: List[FunctionDefinition],
            visualize: bool=False) -> None:
        
        self.functions = functions
        self.visualize = visualize 
    
    def _format_prompt(self, user_input: str) -> str:
        system_guide = (
            "Select one matching function and extract arguments using defined" 
            " types.\nFunctions:\n"
        )

        function_lines = [
            f"- {fn.name}({', '.join(f'{k}: {v.type}' for k, v in fn.parameters.items())}): "
            f"{fn.description}"
            for fn in self.functions
        ]

        functions = "\n".join(function_lines)

        return (
            f"{system_guide}{functions}\n\n"
            f"Request: {user_input}\n"
            "Response:\n"
        )
 

 
    # def _format_prompt(self, user_input: str) -> str:
    #     system_guide = (
    #             "Choose exactly one function that best matches the user's "
    #             "request. Extract its arguments from the request. "
    #             "Do not invent values or numbers. "
    #             "Use the types defined by the function.\n\n"
    #             "Available functions:\n"
    #     )
    #     function_lines = []

    #     for fn in self.functions:
    #         params = []
    #         for p_name, p_def in fn.parameters.items():
    #             params.append(f"{p_name}: {p_def.type}")
    #         params = ", ".join(params)
    #         function_lines.append(f"- {fn.name}({params}): {fn.description}")
    #     functions_str = "\n".join(function_lines)

    #     full_prompt = (
    #             f"{system_guide}{functions_str}\n\n"
    #             f"User Request: {user_input}\nAssistant Response:\n"
    #     )
    #     return full_prompt

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
        # --- PROFILING ACCUMULATORS ---
        t_total_encode = 0.0
        t_total_llm = 0.0
        t_total_sm_allow = 0.0
        t_total_sm_update = 0.0
        t_total_select = 0.0
        
        formated_prompt = self._format_prompt(prompt_txt)
        visualizer: Optional[GenerationVisualizer] = None
        if self.visualize:
            visualizer = GenVisualizer(
                    promt=prompt_txt,
                    max_tokens=max_tokens,
                    vocab_size=len(vocab_mgr.id_to_token)
            )
            visualizer.start()
        try:
            t_start_encode = time.perf_counter()
            input_ids: List[int] = model.encode(formated_prompt).tolist()[0]
            t_total_encode += time.perf_counter() - t_start_encode

            state_machine = JSONStateMachine(
                    prompt_txt,
                    self.functions,
                    model,
                    vocab_mgr,
            )

            token_count = 0

            for _ in range(max_tokens):
                
                t_sm0 = time.perf_counter()
                allowed_ids = state_machine.get_allowed_token_ids()
                t_total_sm_allow += time.perf_counter() - t_sm0
                
                if visualizer:
                    visualizer.update_constraints(allowed_ids, state_machine)
                if state_machine.deterministic_token_ids:
                    deterministic_ids = (
                            state_machine.deterministic_token_ids.copy()
                    )
                    state_machine.deterministic_token_ids.clear() 
                    # Maybe actually remove the line bellow, bc it was
                    # working without it and faster
                    input_ids.extend(state_machine.deterministic_token_ids)
                    for token_id in deterministic_ids:
                        token_text = vocab_mgr.id_to_token[token_id]
                        input_ids.append(token_id)
                        if visualizer:
                            visualizer.update_token(
                            token_id,
                            token_text,
                            deterministic=True,
                        )

                if state_machine.is_complete():
                    if visualizer:
                        visualizer.finish(state_machine)
                    break

                t_llm0 = time.perf_counter()
                logits = model.get_logits_from_input_ids(input_ids)
                t_total_llm += time.perf_counter() - t_llm0
                if visualizer:
                    visualizer.update_logits(logits)

                t_sel0 = time.perf_counter()
                next_token = select_next_token(logits, allowed_ids)
                t_total_select += time.perf_counter() - t_sel0
                token_text = vocab_mgr.id_to_token[next_token]
                token_count += 1
                 
                t_sm1 = time.perf_counter()
                state_machine.update(next_token)
                t_total_sm_update += time.perf_counter() - t_sm1
                input_ids.append(next_token)
                if visualizer:
                    visualizer.update_token(
                            next_token,
                            token_text,
                            deterministic=False
                    )
            else:
                if visualizer:
                    visualizer.status = "Maximum token limit reached"
        finally:
            if visualizer:
                visualizer.stop()
            
        # --- PRINT PROFILING REPORT ---
        # print("\n" + "="*50)
        # print("          PIPELINE PROFILING REPORT")
        # print("="*50)
        # print(f"Total Tokens Generated : {token_count}")
        # print(f"1. Prompt Encoding     : {t_total_encode:.4f} seconds")
        # print(f"2. SM Check Allowed Ids: {t_total_sm_allow:.4f} seconds")
        # print(f"3. LLM Logits Fetch    : {t_total_llm:.4f} seconds")
        # print(f"4. Next Token Select   : {t_total_select:.4f} seconds")
        # print(f"5. SM State Update     : {t_total_sm_update:.4f} seconds")
        # print("="*50 + "\n")

        try:
            parsed_output = json.loads(state_machine.buffer)
            return {
                "prompt": prompt_txt,
                "name": parsed_output.get("name", ""),
                "parameters": parsed_output.get("parameters", {})
            }
        except json.JSONDecodeError as e:
            raise CallMeError(f"Generated output failed JSON parsing: {e}")
