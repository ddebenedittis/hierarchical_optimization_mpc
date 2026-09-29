from pathlib import Path

import numpy as np
import pandas as pd

from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
    plot_distances,
    plot_distances2,
    save_snapshots,
)


def resample(times, values, tt):
    t_new = np.arange(times[0], times.iloc[-1], 0.1)
    values_resampled = np.interp(t_new, tt, values)
    return values_resampled


def main():
    BASE_DIR = Path(__file__).resolve().parent
    df = []
    df_slack = []
    n_c = 1
    dt = 0.05  # 0.012
    s_history = []
    l_hist = [np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0])]
    m_hist = [np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0])]
    folder = 'form'
    if folder == 'nar_gap':
        n_ag = 10
    elif folder == 'rad_switch':
        n_ag = 8
    elif folder == 'form':
        n_ag = 9
    elif folder == 'cov':
        n_ag = 10
    for i in range(n_ag):
        csv_path = BASE_DIR / f'{folder}' / 'velref2' / f'node_{i}_data.csv'
        df_m = pd.read_csv(csv_path)
        df.append(df_m)
        # for i in range(n_ag):
        #     csv_path = BASE_DIR / f'{folder}'/ 'new' /f'node_{i}_data.csv'
        df_m = pd.read_csv(csv_path)
        df_slack.append(df_m)

    for r in range(df[0].shape[0]):
        s = []
        for i in range(n_ag):
            s.append(
                np.array([df[i][f'stateX_{i}'][r], df[i][f'stateY_{i}'][r], 0])
            )  #  df[i][f'stateRHO_{i}'][r]
            for p in range(5):
                # l_hist[p][-1] += (df_slack[i][f'lamda_{p}'][r]/n_ag)
                # m_hist[p][-1] += (df_slack[i][f'mu_{p}'][r]/n_ag)
                l_hist[p][-1] = (
                    df_slack[i][f'lamda_{p}'][r]
                    if df_slack[i][f'lamda_{p}'][r] > l_hist[p][-1]
                    else l_hist[p][-1]
                )
                m_hist[p][-1] = (
                    df_slack[i][f'mu_{p}'][r]
                    if df_slack[i][f'mu_{p}'][r] > m_hist[p][-1]
                    else m_hist[p][-1]
                )
        s_history.append([s, []])
        for p in range(5):
            l_hist[p] = np.append(l_hist[p], 0)
            m_hist[p] = np.append(m_hist[p], 0)

    flags = MultiRobotArtistFlags()
    flags.future_trajectory = False
    flags.voronoi = False
    flags.centroid = True
    flags.legend = True
    # display_animation(
    #     s_history,
    #     s_history,
    #     None,
    #     None,
    #     0.1,
    #     'save',
    #     video_name=f'video_uni.mp4',
    #     x_lim=[-20,20],
    #     y_lim=[-20,20],
    #     flags=flags,
    #     n_c=n_c,
    # )
    # save_snapshots(
    #     s_history,
    #     None,
    #     [[5.5,4.5,2]],#[[0, -8, 5.5], [0, 8, 5.5]],  # [[1.75, 0.28, 0.4]],  # [[3, 3, 0.5]],
    #     0.05,
    #     [16.0 , 33.0],
    #     f'form.pdf',
    #     x_lim=[-5, 20],
    #     y_lim=[-5, 20],
    #     flags=flags,
    # )

    # plot_distances(
    #     s_history,
    #     0.05,  # dt 012
    #     None, #0.6,#0.5, #1.8,  # dmin 0.6
    #     to_obj = True,
    #     form = True,
    # )
    plot_distances2(
        s_history,
        0.05,  # dt 012
        None,  # 0.6,#0.5, #1.8,  # dmin 0.6
        to_obj=True,
        form=True,
        slack=[l_hist, m_hist],
    )


if __name__ == '__main__':
    main()
