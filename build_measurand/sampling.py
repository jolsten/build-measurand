from typing import Literal
from pydantic import BaseModel

class SamplingStrategy(BaseModel):
    window: int
    mode: Literal["mean", "mode"]
