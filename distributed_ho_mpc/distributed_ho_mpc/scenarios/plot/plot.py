from pathlib import Path

import pandas as pd

from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
    plot_distances,
    save_snapshots,
)

BASE_DIR = Path(__file__).resolve().parent
df = []

folder = 'nar_gap'

for i in range(10):
    csv_path = BASE_DIR / f'{folder}' / f'node_{i}_data.csv'
    df_m = pd.read_csv(csv_path)
    df.append(df_m)

s_history = []

for r in range(df[0].shape[0]):
    s = []
    for i in range(10):
        s.append([df[i][f'stateX_{i}'][r], df[i][f'stateY_{i}'][r]])
    s_history.append([s, []])

flags = MultiRobotArtistFlags()
plot_distances(
    s_history,
    0.012,  # dt
    0.5,  # dmin
)
