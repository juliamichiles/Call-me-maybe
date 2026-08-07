from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class ParameterProperty(BaseModel):
    type: str
    description: Optional[str] = None


class ReturnSpec(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str: ParameterProperty]
    returns: ReturnSpec


class PromtTest(BaseModel):
    prompt: str


class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
