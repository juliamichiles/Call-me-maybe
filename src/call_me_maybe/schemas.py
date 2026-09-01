from typing import Dict, Any, Optional

from src.call_me_maybe.errors import CallMeError
try:
    from pydantic import BaseModel
except ImportError as e:  # add more errors 
    raise CallMeError(e) 
# FIXME: Rename type to p_type


class ParameterProperty(BaseModel):
    type: str
    description: Optional[str] = None


class ReturnSpec(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterProperty]
    returns: ReturnSpec


class PromptTest(BaseModel):
    prompt: str


class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
