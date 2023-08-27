import pytest
import numpy as np
from hypothesis import given, assume, strategies as st
from build_measurand.euc import EUC

ST_SCALE_FACTOR = st.floats(min_value=1e-12, max_value=1e12)


@given(st.lists(ST_SCALE_FACTOR, min_size=1, max_size=3))
def test_scale_factor_regex(args):
    csv = ",".join([str(x) for x in args])
    for spec in [f"EUC[{csv}]", f"euc[{csv}]", f"[{csv}]", csv]:
        assert ScaleFactor._REGEX.match(spec)


@given(
    st.integers(min_value=1, max_value=256),
    ST_SCALE_FACTOR,
    ST_SCALE_FACTOR,
    ST_SCALE_FACTOR,
)
def test_parameter_scale_factor(word, db, sf, sb):
    for spec, result in {
        f"{word};u;EUC[{sf}]": (word % 256) * sf,
        f"{word};u;EUC[{db},{sf}]": (word % 256 + db) * sf,
        f"{word};u;EUC[{db},{sf},{sb}]": (word % 256 + db) * sf + sb,
    }.items():
        print(spec)
        p = Parameter(spec)
        assert list(p.build(SAMPLE_DATA)) == pytest.approx([result] * 10)


def test_scale_factor_invalid():
    with pytest.raises(ValueError):
        ScaleFactor("not a valid scale factor")
