from typing import Dict, Any, Optional
from pydantic import BaseModel


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
