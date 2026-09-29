import matplotlib
import pytest

matplotlib.use('Agg')

import distributed_ho_mpc.scenarios.formation_obstacle_omni.settings as st
from distributed_ho_mpc.scenarios.formation_obstacle_omni.network_simulation import main


def test_formation_obstacle_omni():
    try:
        st.n_steps = 2

        main()
    except Exception as e:
        assert False, f'Test failed with exception: {e}'
