from pydantic import ValidationError
from pathlib import Path
from typing import List
import json

from .schemas import FunctionDefinition, PromptTest
from .errors import CallMeError

# FIXME: Either add writing output method here or rename to input or loaders
# FIXME: NOT OOP!! Is that ok?
def load_functions_definition(path: str) -> List[FunctionDefinition]:
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise CallMeError(
                    f"Function definitions file not found: {path}", 
            ) 
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            return [FunctionDefinition.model_validate(item) for item in raw_data]
            
    except (json.JSONDecodeError, ValidationError) as e:
        raise CallMeError(
                f"Invalid function definitions file format: {e}", 
        )

def load_input_prompts(path: str) -> List[PromptTest]:

    try:
        file_path = Path(path)
        if not file_path.exists():
            raise CallMeError(
                    f"Input prompts file not found: {path}", 
            )
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            return [PromptTest.model_validate(item) for item in raw_data]

    except (json.JSONDecodeError, ValidationError) as e:
        raise CallMeError(
                f"Invalid input prompts file format {e}",
        )
