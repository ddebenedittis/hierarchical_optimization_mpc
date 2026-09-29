import numpy as np

from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
)


def resample(times, values):
    t_new = np.arange(times[0], times[-1], 0.1)
    values_resampled = np.interp(t_new, times, values)
    return values_resampled


def main():
    csv_folders = [
        '2026-01-15_16-34-08-dhqp',
        '2026-01-15_16-50-44-dhqp',
        '2026-01-15_16-51-21-dhqp',
        '2026-01-15_16-58-37-dhqp',
        '2026-01-15_17-03-56-dhqp',
    ]

    for csv_folder in csv_folders:
        print(csv_folder)

        data = np.genfromtxt(f'./out/{csv_folder}/traj_data.csv', delimiter=',', names=True)

        n_robots = 5

        times = data['time']
        resampled_data = {}
        for j in range(n_robots):
            resampled_data[f'stateX_{j}'] = resample(times, data[f'stateX_{j}'])
            resampled_data[f'stateY_{j}'] = resample(times, data[f'stateY_{j}'])

        n_timesteps = len(resampled_data[f'stateX_{0}'])

        s_history = [[[]] + [[np.zeros(2) for _ in range(n_robots)]] for _ in range(n_timesteps)]
        for j in range(n_robots):
            for k in range(n_timesteps):
                s_history[k][1][j][0] = resampled_data[f'stateX_{j}'][k]
                s_history[k][1][j][1] = resampled_data[f'stateY_{j}'][k]

        flags = MultiRobotArtistFlags()
        flags.centroid = False
        flags.voronoi = True
        flags.past_trajectory = False
        flags.labels = False
        flags.legend = False
        flags.time = False
        flags.grid = False
        flags.fill_voronoi = True

        display_animation(
            s_history,
            None,
            None,
            0.2,
            'save',
            video_name=f'./vid/{csv_folder}_colored.mp4',
            x_lim=np.array([-1.6, 3.9]),
            y_lim=np.array([-1.15, 1.6]),
            ticks_rotation=90,
            dpi=400,
            flags=flags,
        )

        flags.fill_voronoi = False

        display_animation(
            s_history,
            None,
            None,
            0.1,
            'save',
            video_name=f'./vid/{csv_folder}.mp4',
            x_lim=np.array([-1.6, 3.9]),
            y_lim=np.array([-1.15, 1.6]),
            ticks_rotation=90,
            dpi=400,
            flags=flags,
        )


if __name__ == '__main__':
    main()
