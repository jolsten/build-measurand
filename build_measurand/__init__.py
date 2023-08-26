from typing import List, Optional
from pydantic import BaseModel
from .parameter import Parameter
# from .scale import ScaleFactor
# from .sampling import SamplingStrategy

class Measurand(BaseModel):
    parameter: Parameter
    # scale_factor: Optional[ScaleFactor] = None
    # sampling_strategy: Optional[SampligStrategy] = None
