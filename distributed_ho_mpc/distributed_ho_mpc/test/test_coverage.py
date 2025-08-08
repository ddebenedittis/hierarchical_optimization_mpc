import matplotlib
import pytest

import distributed_ho_mpc.scenarios.coverage.settings as st
from distributed_ho_mpc.scenarios.coverage.network_simulation import main

matplotlib.use('Agg')


def test_coverage_omni():
    try:
        st.n_steps = 2

        main(model_name='omni')
    except Exception as e:
        assert False, f'Test failed with exception: {e}'


def test_coverage_unicycle():
    try:
        st.n_steps = 2

        main(model_name='uni')
    except Exception as e:
        assert False, f'Test failed with exception: {e}'
