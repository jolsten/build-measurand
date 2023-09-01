import pytest
import numpy as np
from hypothesis import given, assume, strategies as st
from build_measurand.euc import ScaleFactorEUC, RE_SCALEFACTOR, make_euc

ST_SCALE_FACTOR = st.floats(min_value=1e-12, max_value=1e12)


@given(st.lists(ST_SCALE_FACTOR, min_size=1, max_size=3))
def test_scale_factor_regex(args):
    csv = ",".join([str(x) for x in args])
    for spec in [f"EUC[{csv}]", f"euc[{csv}]", f"[{csv}]", csv]:
        assert RE_SCALEFACTOR.match(spec)


@given(
    ST_SCALE_FACTOR,
    ST_SCALE_FACTOR,
    ST_SCALE_FACTOR,
)
def test_make_euc_scalefactor(db, sf, sb):
    for spec, result in {
        f"EUC[{sf}]": ScaleFactorEUC(scale_factor=sf),
        f"EUC[{db},{sf}]": ScaleFactorEUC(data_bias=db, scale_factor=sf),
        f"EUC[{db},{sf},{sb}]": ScaleFactorEUC(
            data_bias=db, scale_factor=sf, scaled_bias=sb
        ),
    }.items():
        euc = make_euc(spec)
        assert euc == result


@given(
    st.integers(min_value=0, max_value=255),
    ST_SCALE_FACTOR,
    ST_SCALE_FACTOR,
    ST_SCALE_FACTOR,
)
def test_euc_apply_ndarray(val: int, db: float, sf: float, sb: float):
    euc = ScaleFactorEUC(data_bias=db, scale_factor=sf, scaled_bias=sb)
    data = np.array([val] * 10, dtype="u1")
    result = euc.apply_ndarray(data, 8)
    answer = (data.astype("f8") + db) * sf + sb
    assert result.tolist() == answer.tolist()


def test_scale_factor_invalid():
    with pytest.raises(ValueError):
        make_euc("not a valid scale factor")
