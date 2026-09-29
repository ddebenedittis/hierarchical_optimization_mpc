from pathlib import Path

import numpy as np
import pandas as pd

from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
    plot_distances,
    save_snapshots,
)


def resample(times, values, tt):
    t_new = np.arange(times[0], times.iloc[-1], 0.1)
    values_resampled = np.interp(t_new, tt, values)
    return values_resampled


def main():
    BASE_DIR = Path(__file__).resolve().parent
    df = []
    n_c = 1
    n_robots = 5
    folder = 'mpc_data'
    s_history = []
    df_real = pd.read_csv(BASE_DIR / f'{folder}' / 'rad' / f'traj_data.csv')

    times = df_real['time']

    for r in range(15, df_real['time'].shape[0]):
        s = []
        for i in range(n_robots):
            s.append(
                np.array(
                    [
                        df_real[f'stateX_{i}'][r],
                        df_real[f'stateY_{i}'][r],
                        df_real[f'stateRHO_{i}'][r],
                    ]
                )
            )
        s_history.append([s, []])

    s_history_all = []

    flags = MultiRobotArtistFlags()
    flags.future_trajectory = False
    flags.voronoi = False
    flags.centroid = False
    display_animation(
        s_history,
        s_history_all,
        None,
        None,  # [[1, 0.25, 0.5]],
        0.1,
        'none',
        video_name=f'video.mp4',
        x_lim=[-1, 3],
        y_lim=[-1.4, 2],
        flags=flags,
        n_c=n_c,
    )
    # save_snapshots(
    #     s_history,
    #     None,
    #     None,  # [[1.75, 0.28, 0.4]],  # [[3, 3, 0.5]],
    #     0.05,
    #     [(663) * 0.05],
    #     f'nar_gap_half.pdf',
    #     x_lim=[-7, 7],
    #     y_lim=[-5, 5],
    #     flags=flags,
    # )

    plot_distances(
        s_history,
        0.1,  # dt 012
        0.5,  # dmin
        to_obj=False,
        form=False,
    )


"""def main():
    BASE_DIR = Path(__file__).resolve().parent
    df = []
    n_c = 4
    n_robots = 4
    folder = 'mpc_data'
    for i in range(n_robots):
        csv_path = BASE_DIR / f'{folder}' / f'mpc_data_{i}.csv'
        df_m = pd.read_csv(csv_path)
        df.append(df_m)
    s_history = []
    df_real =  pd.read_csv(BASE_DIR / f'{folder}'/'obst'/f'real_position.csv')
        
    times = df_real['time']
    resampled_data = {}
    for nn in range(n_robots):
        resampled_data['time'] = np.arange(times[0], times.iloc[-1], 0.1)
        resampled_data[f'stateX_{nn}'] = resample(times, df_real[f'stateX_{nn}'], times)
        resampled_data[f'stateY_{nn}'] = resample(times, df_real[f'stateY_{nn}'], times)
        resampled_data[f'stateRHO_{nn}'] = resample(times, df_real[f'stateRHO_{nn}'], times)
        
    for nn ,data in enumerate(df):
        for k in range(n_c):
            resampled_data[f'stateX_{nn}_k{k}'] = resample(times, data[f'0_sx_k{k}'][3:], data['time'][3:])
            resampled_data[f'stateY_{nn}_k{k}'] = resample(times, data[f'0_sy_k{k}'][3:], data['time'][3:])
    
    resampled_data = pd.DataFrame.from_dict(resampled_data)

    for r in range(15,resampled_data['time'].shape[0]):
        s = []
        for i in range(n_robots):
            s.append(np.array([resampled_data[f'stateX_{i}'][r], resampled_data[f'stateY_{i}'][r], resampled_data[f'stateRHO_{i}'][r]]))
        s_history.append([s, []])
    
    s_history_all = []
    for r in range(15,resampled_data['time'].shape[0]):
        s = []
        for i in range(n_robots):
            s_k = []
            s_k.append(np.array([resampled_data[f'stateX_{i}'][r], resampled_data[f'stateY_{i}'][r], resampled_data[f'stateRHO_{i}'][r]]))
            for k in range(1,n_c):
                s_k.append(np.array([resampled_data[f'stateX_{i}_k{k}'][r], resampled_data[f'stateY_{i}_k{k}'][r]]))
            s.append(s_k)
        s_history_all.append([[],s])



    flags = MultiRobotArtistFlags()
    flags.future_trajectory = True
    flags.voronoi = False
    flags.centroid = False
    display_animation(
        s_history,
        s_history_all,
        None,
        None, #[[1, 0.25, 0.5]],
        0.1,
        'none',
        video_name=f'video.mp4',
        x_lim=[-1, 3],
        y_lim=[-1.4, 2],
        flags=flags,
        n_c=n_c,
    )
    # save_snapshots(
    #     s_history,
    #     None,
    #     None,  # [[1.75, 0.28, 0.4]],  # [[3, 3, 0.5]],
    #     0.05,
    #     [(663) * 0.05],
    #     f'nar_gap_half.pdf',
    #     x_lim=[-7, 7],
    #     y_lim=[-5, 5],
    #     flags=flags,
    # )


    plot_distances(
        s_history,
        0.1,  # dt 012
        0.5,  # dmin
        to_obj=True,
        form=False
    )"""
if __name__ == '__main__':
    main()
