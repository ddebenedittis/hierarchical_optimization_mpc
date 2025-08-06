import matplotlib
import pytest

matplotlib.use('Agg')


def test_coverage_omni():
    import distributed_ho_mpc.scenarios.coverage_omni.settings as st
    from distributed_ho_mpc.scenarios.coverage_omni.network_simulation import main

    try:
        st.n_steps = 2

        main()
    except Exception as e:
        assert False, f'Test failed with exception: {e}'


def test_coverage_unicycle():
    import distributed_ho_mpc.scenarios.coverage_unicycle.settings as st
    from distributed_ho_mpc.scenarios.coverage_unicycle.network_simulation import main

    try:
        st.n_steps = 2

        main()
    except Exception as e:
        assert False, f'Test failed with exception: {e}'
