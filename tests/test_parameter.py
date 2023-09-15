from typing import List
import pytest
from hypothesis import given, assume, strategies as st
from build_measurand.parameter import (
    Component,
    Parameter,
    make_parameter,
)
from . import strategies as cst
from .conftest import ARRAY_SIZE, SAMPLE_DATA


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
def test_parameter_components(spec, components):
    comps = (Component.from_spec(c) for c in components)
    param = make_parameter(spec)
    print(comps, param.components)
    assert list(param.components) == list(comps)


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
def test_parameter_size(spec, word_size, size):
    r = make_parameter(spec, word_size=word_size)
    assert r.size == size


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
def test_build_parameter_8bit(spec, result):
    p = make_parameter(spec, word_size=8)
    print("spec =", spec)
    print(f"result = {result:0{p.size}b}")
    out = p.build(SAMPLE_DATA[8])
    assert list(out) == list([result] * ARRAY_SIZE)


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
def test_build_parameter_12bit(spec, result):
    r = Parameter.from_spec(spec, word_size=12)
    print("spec =", spec)
    print(f"result = 0b{result:0{r.size}b}")
    assert list(r.build(SAMPLE_DATA[12])) == [result] * ARRAY_SIZE


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


@given(cst.parameter_spec())
def test_parameter_eq(spec):
    print("spec =", spec)
    p1 = make_parameter(spec)
    p2 = make_parameter(spec)
    print("param 1:", p1)
    print("param 2:", p2)
    assert p1 == p2


@given(components_spec(), components_spec())
def test_parameter_ne(spec1, spec2):
    assume(spec1 != spec2)
    print(spec1, spec2)
    p1 = make_parameter(spec1)
    p2 = make_parameter(spec2)
    print(p1, p2, p1 == p2, p1 != p2)
    assert p1 != p2


@given(st.integers(min_value=1, max_value=4096))
def test_parameter_12bit(word):
    p = make_parameter(f"{word}")
    assert list(p.build(SAMPLE_DATA[12])) == [word % 2**12] * ARRAY_SIZE
