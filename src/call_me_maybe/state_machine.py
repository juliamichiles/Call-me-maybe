from .schemas import FunctionDefinition
from .vocabulary import VocabularyManager
from typing import Optional, List, Dict, Any, Set
from enum import Enum, auto

def state_prototype(token_id: int) -> 



class ConstraintType(Enum):
    EXTRACT_STRING = auto()
    CHOICE = auto()
    NUMBER = auto()
    STRING_VAL = auto()
    BOOLEAN = auto()


class JSONStateMachine:
    """Tracks current state during token generation and determines allowed 
            next tokens.
    """
    def __init__(
            self,
            prompt_txt: str,
            funcs: List[FunctionDefinition],
            vocab_mgr: VocabManager
        ) -> None:
        self.prompt_txt = prompt_txt
        self.functions = funcs
        self.vocab_mgr = vocab_mgr
        self.selected_function: Optional[FunctionDefinition] = None
        self.current_buffer: str = ""
        

    # FIXME: Does this fuction really belongs here? If not, MOVE IT!!!
    def format_prompt(
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
