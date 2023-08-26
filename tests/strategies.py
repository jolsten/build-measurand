from hypothesis import strategies as st

valid_parameter_sizes = st.integers(min_value=1, max_value=64)

bit_positions = st.integers(min_value=0, max_value=15)
