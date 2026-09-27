from typing import Dict, Any, Optional

from src.call_me_maybe.errors import CallMeError
try:
    from pydantic import BaseModel
except (ModuleNotFoundError, ImportError) as e:
    raise CallMeError(e)
# FIXME: Rename type to p_type


class ParameterProperty(BaseModel):
    """Define a function parameter's type and optional description.
        Fields:
            type: Data type of the parameter.
            description: Optional description of the parameter.
    """
    type: str
    description: Optional[str] = None


class ReturnSpec(BaseModel):
    """Define the return type of a function.
        Fields:
            type: Data type returned by the function.
    """
    type: str


class FunctionDefinition(BaseModel):
    """Define a callable function and its parameters.
        Fields:
            name: Name of the function.
            description: Description of the function's purpose.
            parameters: Mapping of parameter names to their definitions.
            returns: Specification of the function's return type.
    """
    name: str
    description: str
    parameters: Dict[str, ParameterProperty]
    returns: ReturnSpec


class PromptTest(BaseModel):
    """Define a test prompt for function-call generation.
        Fields:
            prompt: User request to use as a test input.
    """
    prompt: str


class FunctionCallOutput(BaseModel):
    """Represent a generated function call and its arguments.
        Fields:
            prompt: User request that produced the function call.
            name: Name of the selected function.
            parameters: Arguments generated for the function call.
    """
    prompt: str
    name: str
    parameters: Dict[str, Any]
