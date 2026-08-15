from pydantic import ValidationError
from pathlib import Path
from typing import List
import json
import sys

from .schemas import FunctionDefinition, PromptTest


# FIXME: Either add writing output method here or rename to input or loaders
# FIXME: NOT OOP!! Is that ok?
# TODO: raise custom errors instead of printing them here directly?
def load_functions_definition(path: str) -> List[FunctionDefinition]:
    try:
        file_path = Path(path)
        if not file_path.exists():
            print(
                    f"Error: Function definitions file not found: {path}", 
                    file=sys.stderr
            )
            sys.exit(1)
            
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            return [FunctionDefinition.model_validate(item) for item in raw_data]
            
    except (json.JSONDecodeError, ValidationError) as e:
        print(
                f"Error: Invalid function definitions file format: {e}", 
                file=sys.stderr
        )
        sys.exit(1)

def load_input_prompts(path: str) -> List[PromtTest]:

    try:
        file_path = Path(path)
        if not file_path.exists():
            print(
                    f"Error: Input prompts file not found: {path}", 
                    file=sys.stderr
            )
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            return [PromtTest.model_validate(item) for item in raw_data]

    except (json.JSONDecodeError, ValidationError) as e:
        print(
                f"Error: Invalid input prompts file format {e}",
                file=sys.sys.stderr
        )
        sys.exit(1)
