import pytest
from hypothesis import given, assume, strategies as st
from build_measurand.parameter import make_parameter
from build_measurand.measurand import make_measurand
from . import strategies as cst
from .conftest import ARRAY_SIZE, SAMPLE_NDARRAY, SAMPLE_PAARRAY
from .cases import Example, parameter_test_cases, component_test_cases


@pytest.mark.parametrize("case", parameter_test_cases)
def test_measurand_size(case: Example):
    r = make_measurand(case.spec, word_size=case.word_size)
    assert r.size == case.size


@pytest.mark.parametrize("case", parameter_test_cases)
class TestBuildMeasurand:
    def test_build_ndarray(self, case: Example):
        m = make_measurand(
            case.spec, word_size=case.word_size, one_based=case.one_based
        )
        print("spec =", case.spec, "result =", f"{case.result:0{m.size}b}")
        out = m.build_ndarray(SAMPLE_NDARRAY[case.word_size])
        assert list(out) == list([case.result] * ARRAY_SIZE)

    def test_build_paarray(self, case: Example):
        m = make_measurand(
            case.spec, word_size=case.word_size, one_based=case.one_based
        )
        print("spec =", case.spec, "result =", f"{case.result:0{m.size}b}")
        out = m.build_paarray(SAMPLE_PAARRAY[case.word_size])
        assert out.to_pylist() == list([case.result] * ARRAY_SIZE)
