import matplotlib
import pytest

matplotlib.use('Agg')

import distributed_ho_mpc.scenarios.radial_switching.settings as st
from distributed_ho_mpc.scenarios.radial_switching.network_simulation import main


def test_radial_switching():
    try:
        st.n_steps = 2

        main()
    except Exception as e:
        assert False, f'Test failed with exception: {e}'
