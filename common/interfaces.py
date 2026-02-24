
from pydantic import BaseModel, Field
from typing import List

class AlertPayload(BaseModel):
    request_id: str
    message: str
    trace: List[str] = Field(default_factory=list)
    source: str
    
