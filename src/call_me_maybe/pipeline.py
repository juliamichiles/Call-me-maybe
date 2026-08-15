from typing import List

from .schemas import FunctionDefinition 

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
