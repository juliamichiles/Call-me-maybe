from pathlib import Path
from typing import List
import json

from .schemas import FunctionDefinition, PromptTest
from .errors import CallMeError

try:
    from pydantic import ValidationError
except (ModuleNotFoundError, ImportError) as e:
    raise CallMeError(e)


def load_functions_definition(path: str) -> List[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.
        Args: path: Path to the JSON file containing function definitions.
        Returns: A list of validated function definitions.
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise CallMeError(
                    f"Function definitions file not found: {path}",
            )
        with open(file_path, "r", encoding="utf-8") as f:
            r_data = json.load(f)
            return [FunctionDefinition.model_validate(item) for item in r_data]

    except (json.JSONDecodeError, ValidationError, PermissionError) as e:
        raise CallMeError(
                f"Invalid function definitions file format: {e}",
        )


def load_input_prompts(path: str) -> List[PromptTest]:
    """Load and validate input prompts from a JSON file.
        Args: path: Path to the JSON file containing input prompts.
        Returns: A list of validated input prompts.
    """
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
