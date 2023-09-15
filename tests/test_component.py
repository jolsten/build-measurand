import pytest
from hypothesis import given
import hypothesis.strategies as st
from build_measurand.component import RE_COMPONENT, make_component
from . import strategies as cst
from .conftest import ARRAY_SIZE, SAMPLE_DATA


@given(
    cst.component_spec(),
    st.booleans(),
)
def test_re_component(word_spec, reversed):
    spec1 = f'{word_spec}{"R" if reversed else ""}'
    print(spec1)
    assert RE_COMPONENT.match(spec1)


@given(
    st.integers(min_value=1, max_value=1023),
    st.tuples(
        st.integers(min_value=0, max_value=7),
        st.integers(min_value=0, max_value=7),
    ),
    st.booleans(),
)
def test_re_component_with_bits(word, bits, reversed):
    a, b = bits
    if a == b:
        bits = a
    else:
        bits = f"{a}-{b}"
    spec2 = f'{word}:{bits}{"R" if reversed else ""}'
    print(spec2)
    assert RE_COMPONENT.match(spec2)


@given(cst.word(one_based=True))
def test_component_1_based_byte(word):
    spec = f"{word}"
    c = make_component(spec)
    assert c.word == word - 1
    assert c.mask == None
    assert c.shift == 0


@given(cst.word(one_based=False))
def test_component_0_based_byte(word):
    spec = f"{word}"
    c = make_component(spec, one_based=False)
    assert c.word == word
    assert c.mask == None
    assert c.shift == 0


@given(
    cst.word(one_based=True),
    cst.bit(one_based=True, word_size=16),
)
def test_component_1_based_bit(word, bit):
    spec = f"{word}:{bit}"
    print(spec)
    c = make_component(spec, one_based=True)
    print("c.word =", c.word)
    print("c.mask =", c.mask)
    assert c.word == word - 1
    assert c.mask == 2 ** (bit - 1)


@given(
    cst.word(one_based=False),
    cst.bit(one_based=False, word_size=16),
)
def test_component_0_based_bit(word, bit):
    spec = f"{word}:{bit}"
    c = make_component(spec, one_based=False)
    assert c.word == word
    assert c.mask == 2**bit


@pytest.mark.parametrize(
    "spec, word_size, size",
    [
        ("1", 8, 8),
        ("1", 10, 10),
        ("1:1-4", 8, 4),
        ("1:1-2", 8, 2),
    ],
)
def test_component_size(spec, word_size, size):
    c = make_component(spec, word_size=word_size)
    assert c.size == size


@pytest.mark.parametrize(
    "spec",
    [
        "1:not-valid",
        "1=2",
        "a",
    ],
)
def test_component_invalid_spec(spec):
    with pytest.raises(ValueError):
        make_component(spec)


@pytest.mark.parametrize(
    "spec, result",
    [
        ("1", 1),
        ("128", 128),
        ("255", 255),
        ("1:1", 1),
        ("256:1", 0),
        ("170:1-4", 0xA),
        ("170:5-8", 0xA),
        ("1R", 128),
        ("128R", 1),
        ("170:1-4R", 0x5),
        ("170:5-8R", 0x5),
    ],
)
def test_component_build(spec, result):
    c = make_component(spec)
    assert list(c.build(SAMPLE_DATA[8])) == list([result] * ARRAY_SIZE)


# @given(cst.component_spec(), st.sampled_from(SAMPLE_DATA.keys()))
# class TestComponentBuild8:
#     WORD_SIZE = 8

#     @property
#     def sample_data(self):
#         return SAMPLE_DATA[self.word_size]

#     def test_build(self, spec, result):
#         c = make_component(spec)
#         assert list(c.build(SAMPLE_DATA[8])) == list([result] * ARRAY_SIZE)
