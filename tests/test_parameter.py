import pytest
import numpy as np
from hypothesis import given, assume, strategies as st
from build_measurand.parameter import (
    Component,
    Parameter,
    make_parameter,
    _expand_component_range,
)


@pytest.mark.parametrize(
    "spec, components",
    [
        ("1+2", ["1", "2"]),
        ("2+1", ["2", "1"]),
        ("1-4", ["1", "2", "3", "4"]),
        ("4-1", ["4", "3", "2", "1"]),
        ("1:1-4+2:5-8", ["1:1-4", "2:5-8"]),
        ("1:1-4+2:5-8R", ["1:1-4", "2:5-8R"]),
    ],
)
def test_raw_param_components(spec, components):
    comps = [Component.from_spec(c) for c in components]
    assert make_parameter(spec).components == comps


@pytest.mark.parametrize(
    "spec, word_size, size",
    [
        ("1", 8, 8),
        ("1", 10, 10),
        ("1+2", 8, 16),
        ("1:1-4+2:5-8", 8, 8),
        ("1-4", 8, 32),
        ("1-3", 10, 30),
    ],
)
def test_rawparam_size(spec, word_size, size):
    r = make_parameter(spec, word_size=word_size)
    assert r.size == size


SAMPLE_DATA = np.array([np.arange(start=1, stop=257, dtype="u1")] * 10)
SAMPLE_DATA_12 = np.array([np.arange(start=1, stop=4097, dtype="u2")] * 10)


@pytest.mark.parametrize(
    "spec, result",
    [
        ("1", 0x01),
        ("255", 0xFF),
        ("1+2", 0x0102),
        ("255+255", 0xFFFF),
        ("1+256", 0x0100),
        ("256+1", 0x0001),
        ("1-4", 0x01020304),
        ("4-1", 0x04030201),
        ("1R", 0x80),
        ("10:1-4R", 0x5),
        ("10:5-8R", 0x0),
        ("170:1-4R", 0x5),
        ("170:5-8R", 0x5),
    ],
)
def test_build_rawparam(spec, result):
    r = Parameter.from_spec(spec, word_size=8)
    print("spec =", spec)
    print(f"result = {result:0{r.size}b}")
    assert r.build(SAMPLE_DATA)[0] == result


@pytest.mark.parametrize(
    "spec, result",
    [
        ("1", 0x001),
        ("1R", 0x800),
        ("4095:1-4", 0x00F),
        ("4095:5-8", 0x00F),
        ("4095:9-12", 0x00F),
        ("4095:1-8", 0x0FF),
        ("4095:5-12", 0x0FF),
    ],
)
def test_build_rawparam_10bit(spec, result):
    r = Parameter(spec, word_size=12)
    print("spec =", spec)
    print(f"result = 0b{result:0{r.size}b}")
    assert r.build(SAMPLE_DATA_12)[0] == result


##############
# Parameters #
##############


@st.composite
def components_spec(draw, max_words=256):
    num_components = draw(st.integers(min_value=1, max_value=8))
    start = draw(st.integers(min_value=1, max_value=max_words))
    stop = start + num_components - 1
    assume(stop <= max_words)
    return "+".join([str(x) for x in list(range(start, stop + 1))])


@given(components_spec())
def test_parameter_eq(spec):
    spec = f"{spec};2c"
    assert Parameter(spec) == Parameter(spec)


@given(components_spec(), components_spec())
def test_parameter_ne(spec1, spec2):
    assume(spec1 != spec2)
    print(spec1, spec2)
    spec1, spec2 = f"{spec1};2c", f"{spec2};2c"
    p1 = Parameter(spec1)
    p2 = Parameter(spec2)
    print(p1, p2, p1 == p2, p1 != p2)
    assert p1 != p2


@given(st.integers(min_value=1, max_value=256))
def test_parameter(word):
    p = Parameter(f"{word}")
    assert list(p.build(SAMPLE_DATA)) == [word % 256] * 10


@given(st.integers(min_value=1, max_value=4096))
def test_parameter_12bit(word):
    p = Parameter(f"{word}")
    assert list(p.build(SAMPLE_DATA_12)) == [word % 256] * 10


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
def test_parameter_8bit_2c(uint, int):
    p = Parameter(f"{uint};2c")
    assert p.build(SAMPLE_DATA)[0] == int
