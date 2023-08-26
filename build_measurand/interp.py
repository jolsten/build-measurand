from abc import ABC, abstractmethod
import numpy as np
from .factory import ObjectFactory
from .types.ufunc import (
    onescomp,
    twoscomp,
    milstd1750a32,
    milstd1750a48,
    ti32,
    ti40,
    ibm32,
    ibm64,
    dec32,
    dec64,
    dec64g,
)


class InterpStrategy(ABC):
    @abstractmethod
    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class InvalidInterpType(ValueError):
    def __init__(self, interp_spec: str) -> None:
        self.msg = f'"{interp_spec}" is not a valid interpretation specification.'

    def __str__(self) -> str:
        return self.msg


class InvalidInterpSize(ValueError):
    def __init__(
        self, cls: InterpStrategy, expected_size: int, received_size: int
    ) -> None:
        self.msg = f"{cls.__name__} expects a word with size {expected_size}"
        f"but was provided a word with size {received_size}"

    def __str__(self) -> str:
        return self.msg


interp = ObjectFactory()


@interp.register("u")
class Unsigned(InterpStrategy):
    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        return data


@interp.register("1c")
class OnesComplement(InterpStrategy):
    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        return onescomp(data, bits)


@interp.register("2c")
class TwosComplement(InterpStrategy):
    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        return twoscomp(data, bits)


@interp.register("ieee16")
class IEEE16(InterpStrategy):
    SIZE = 16

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return data.view(">f2")


@interp.register("ieee32")
class IEEE32(InterpStrategy):
    __name__ = "ieee32"
    SIZE = 32

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return data.view(">f4")


@interp.register("ieee64")
class IEEE64(InterpStrategy):
    SIZE = 64

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return data.view(">f8")


@interp.register("1750a32")
class MilStd1750A32(InterpStrategy):
    SIZE = 32

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return milstd1750a32(data)


@interp.register("1750a48")
class MilStd1750A48(InterpStrategy):
    SIZE = 48

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return milstd1750a48(data)


@interp.register("ti32")
class TI32(InterpStrategy):
    SIZE = 32

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return ti32(data)


@interp.register("ti40")
class TI40(InterpStrategy):
    SIZE = 40

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return ti40(data)


@interp.register("ibm32")
class IBM32(InterpStrategy):
    SIZE = 32

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return ibm32(data)


@interp.register("ibm64")
class IBM64(InterpStrategy):
    SIZE = 64

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return ibm64(data)


class DEC32(InterpStrategy):
    SIZE = 32

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return dec32(data)


class DEC64(InterpStrategy):
    SIZE = 64

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return dec64(data)


class DEC64G(InterpStrategy):
    SIZE = 64

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return dec64g(data)
