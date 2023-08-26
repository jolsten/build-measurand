import pytest
from hypothesis import given
import hypothesis.strategies as st
from build_measurand.component import Component, RE_COMPONENT, make_component


@given(
    st.integers(min_value=1, max_value=1023),
    st.booleans(),
)
def test_re_component(word, reversed):
    spec1 = f'{word}{"R" if reversed else ""}'
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


@given(st.integers(min_value=1, max_value=1024))
def test_component_1_based_byte(word):
    spec = f"{word}"
    c = make_component(spec)
    assert c.word == word - 1
    assert c.mask == None
    assert c.shift == 0


@given(st.integers(min_value=0, max_value=1023))
def test_component_0_based_byte(word):
    spec = f"{word}"
    c = make_component(spec, one_based=False)
    assert c.word == word
    assert c.mask == None
    assert c.shift == 0


@given(
    st.integers(min_value=1, max_value=1024),
    st.integers(min_value=1, max_value=8),
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
    st.integers(min_value=0, max_value=1023),
    st.integers(min_value=0, max_value=7),
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
