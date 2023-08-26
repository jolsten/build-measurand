import pytest
from hypothesis import given, assume, strategies as st
import numpy as np
from build_measurand.utils import _size_to_uint
from build_measurand.interp import (
    interp,
    Unsigned,
    OnesComplement,
    TwosComplement,
    IEEE16,
    IEEE32,
    IEEE64,
    MilStd1750A32,
    MilStd1750A48,
    TI32,
    TI40,
    InvalidInterpSize,
    InvalidInterpType,
)


@st.composite
def uint(draw, size: int):
    return draw(st.integers(min_value=0, max_value=2**size - 1))


@st.composite
def uint_and_size(draw, max_size: int = 64):
    size = draw(st.integers(min_value=1, max_value=max_size))
    value = draw(uint(size))
    return value, size


@given(uint_and_size())
def test_unsigned(things):
    uint, size = things
    dtype = _size_to_uint(size)
    data = np.array([uint] * 10, dtype=dtype)
    result = Unsigned().apply_ndarray(data, size)
    assert list(result) == [uint] * 10


@given(uint_and_size())
def test_onescomp(things):
    uint, size = things
    dtype = _size_to_uint(size)
    data = np.array([uint] * 10, dtype=dtype)
    result = OnesComplement().apply_ndarray(data, size)
    assert result.shape[0] == 10


@given(uint_and_size())
def test_twoscomp(things):
    uint, size = things
    dtype = _size_to_uint(size)
    data = np.array([uint] * 10, dtype=dtype)
    result = TwosComplement().apply_ndarray(data, size)
    assert result.shape[0] == 10


@given(uint(16))
def test_ieee16(uint):
    data = np.array([uint] * 10, dtype="u2")
    result = IEEE16().apply_ndarray(data, 16)
    assert result.shape[0] == 10


@given(uint(32))
def test_ieee32(uint):
    data = np.array([uint] * 10, dtype="u4")
    result = IEEE32().apply_ndarray(data, 32)
    assert result.shape[0] == 10


@given(uint(64))
def test_ieee64(uint):
    data = np.array([uint] * 10, dtype="u8")
    result = IEEE64().apply_ndarray(data, 64)
    assert result.shape[0] == 10


@given(uint(32))
def test_1750a32(uint):
    data = np.array([uint] * 10, dtype="u4")
    result = MilStd1750A32().apply_ndarray(data, 32)
    assert result.shape[0] == 10


@given(uint(48))
def test_1750a48(uint):
    data = np.array([uint] * 10, dtype="u8")
    result = MilStd1750A48().apply_ndarray(data, 48)
    assert result.shape[0] == 10


@given(uint(32))
def test_ti32(uint):
    data = np.array([uint] * 10, dtype="u4")
    result = TI32().apply_ndarray(data, 32)
    assert result.shape[0] == 10


@given(uint(40))
def test_ti40(uint):
    data = np.array([uint] * 10, dtype="u8")
    result = TI40().apply_ndarray(data, 40)
    assert result.shape[0] == 10


@pytest.mark.parametrize(
    "uint, int",
    [
        (0b000, 0),
        (0b001, 1),
        (0b010, 2),
        (0b011, 3),
        (0b100, -3),
        (0b101, -2),
        (0b110, -1),
        (0b111, 0),
    ],
)
def test_onescomp_3bit(uint, int):
    data = np.array([uint] * 10, dtype="u1")
    strategy = OnesComplement()
    assert list(strategy.apply_ndarray(data, 3)) == [int] * 10


@pytest.mark.parametrize(
    "uint, int",
    [
        (0b00000000, 0),
        (0b00000001, 1),
        (0b00000010, 2),
        (0b01111110, 126),
        (0b01111111, 127),
        (0b10000000, -127),
        (0b10000001, -126),
        (0b11111101, -2),
        (0b11111110, -1),
        (0b11111111, 0),
    ],
)
def test_onescomp_8bit(uint, int):
    data = np.array([uint] * 10, dtype="u1")
    strategy = OnesComplement()
    assert list(strategy.apply_ndarray(data, 8)) == [int] * 10


@pytest.mark.parametrize(
    "uint, int",
    [
        (0b000, 0),
        (0b001, 1),
        (0b010, 2),
        (0b011, 3),
        (0b100, -4),
        (0b101, -3),
        (0b110, -2),
        (0b111, -1),
    ],
)
def test_twoscomp_3bit(uint, int):
    data = np.array([uint] * 10, dtype="u1")
    strategy = TwosComplement()
    assert list(strategy.apply_ndarray(data, 3)) == [int] * 10


@pytest.mark.parametrize(
    "uint, int",
    [
        (0b00000000, 0),
        (0b00000001, 1),
        (0b00000010, 2),
        (0b01111110, 126),
        (0b01111111, 127),
        (0b10000000, -128),
        (0b10000001, -127),
        (0b10000010, -126),
        (0b11111110, -2),
        (0b11111111, -1),
    ],
)
def test_twoscomp_8bit(uint, int):
    data = np.array([uint] * 10, dtype="u1")
    strategy = TwosComplement()
    assert list(strategy.apply_ndarray(data, 8)) == [int] * 10


@pytest.mark.parametrize(
    "uint, value",
    [
        (0x7FFFFF7F, 0.9999998 * 2**127),
        (0x4000007F, 0.5 * 2**127),
        (0x50000004, 0.625 * 2**4),
        (0x40000001, 0.5 * 2**1),
        (0x40000000, 0.5 * 2**0),
        (0x400000FF, 0.5 * 2**-1),
        (0x40000080, 0.5 * 2**-128),
        (0x00000000, 0.0 * 2**0),
        (0x80000000, -1.0 * 2**0),
        (0xBFFFFF80, -0.5000001 * 2**-128),
        (0x9FFFFF04, -0.7500001 * 2**4),
    ],
)
def test_1750a32(uint, value):
    data = np.array([uint] * 10, dtype="u4")
    strategy = MilStd1750A32()
    assert list(strategy.apply_ndarray(data, 32)) == pytest.approx([value] * 10)


@pytest.mark.parametrize(
    "uint, value",
    [
        (0x4000007F0000, 0.5 * 2**127),
        (0x400000000000, 0.5 * 2**0),
        (0x400000FF0000, 0.5 * 2**-1),
        (0x400000800000, 0.5 * 2**-128),
        (0x8000007F0000, -1.0 * 2**127),
        (0x800000000000, -1.0 * 2**0),
        (0x800000FF0000, -1.0 * 2**-1),
        (0x800000800000, -1.0 * 2**-128),
        (0x000000000000, 0.0 * 2**0),
        (0xA00000FF0000, -0.75 * 2**-1),
    ],
)
def test_1750a48(uint, value):
    data = np.array([uint] * 10, dtype="u8")
    strategy = MilStd1750A48()
    assert list(strategy.apply_ndarray(data, 48)) == pytest.approx([value] * 10)


INTERP_SPECS = list(interp._registry.keys())


@given(st.sampled_from(INTERP_SPECS))
def test_equality(strategy):
    i1, i2 = interp.create(strategy), interp.create(strategy)
    assert i1 == i2


@given(st.sampled_from(INTERP_SPECS), st.sampled_from(INTERP_SPECS))
def test_not_equal(s1, s2):
    assume(s1 != s2)
    i1, i2 = interp.create(s1), interp.create(s2)
    assert i1 != i2


###################
# Exception Tests #
###################


@given(st.integers())
def test_invalid_size_ieee32(size):
    assume(size != 32)
    with pytest.raises(InvalidInterpSize):
        IEEE32().apply_ndarray(np.array([0], dtype="u4"), size)
