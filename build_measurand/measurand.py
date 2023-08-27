from typing import List, Optional
from pydantic import BaseModel, Field
from .parameter import Parameter, make_parameter
from .interp import Interp, make_interp
from .euc import EUC, make_euc

# from .scale import ScaleFactor
# from .sampling import SamplingStrategy


class Measurand(BaseModel):
    parameter: Parameter
    interp: Optional[Interp] = None
    euc: Optional[EUC] = None
    # sampling: Optional[Sampling] = None


def make_measurand(spec: str) -> Measurand:
    parts = spec.split(";")
    parameter = make_parameter(parts[0])

    if len(parts) >= 1:
        interp = make_interp(parts[1])

    if len(parts) >= 2:
        euc = make_euc(parts[2])

    return Measurand(parameter=parameter, interp=interp, euc=euc)
