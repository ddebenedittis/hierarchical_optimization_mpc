from dataclasses import dataclass, field
import itertools
from typing import Any, Optional

from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from hierarchical_optimization_mpc.utils.disp_het_multi_rob import (
    gen_arrow_head_marker,
    init_matplotlib,
    Player,
)
from hierarchical_optimization_mpc.utils.robot_models import RobCont
from hierarchical_optimization_mpc.voronoi_task import BoundedVoronoi


@dataclass
class MultiRobotArtists:
    centroid: Optional[Any] = None
    goals: Optional[Any] = None
    obstacles: Optional[Any] = None
    past_trajectory: Optional[Any] = None
    robots: RobCont = field(default_factory=RobCont)
    voronoi: Optional[Any] = None

# ================================= Animation ================================ #

class Animation():    
    def __init__(
        self,
        data,
        artist_flags: MultiRobotArtists,
        goals,
        obstacles,
        ax,
        dt: float
    ) -> None:
        self.textsize = init_matplotlib()
        
        self.n_history = np.inf
        
        self.data = data
        self.goals = goals
        self.obstacles = obstacles
        self.ax = ax
        self.dt = dt
        
        self.x_lim = [-20., 20.]
        self.y_lim = [-20., 20.]
        
        self.n_robots = [len(data_i) for data_i in data[0]]
        
        self.artists = None
        
        self.artist_flags = artist_flags
        self.show_trajectory = True
        self.show_voronoi = True
        
    
    # ================================= Init ================================= #
    
    def init(self):
        self.ax.clear()
        
        self.ax.set_aspect('equal', 'box')
        
        self.ax.set(xlim=self.x_lim, ylim=self.y_lim, xlabel='$x$ [$m$]', ylabel='$y$ [$m$]')
        
        # ====================== Initialize The Artists ====================== #
        
        self.artists = MultiRobotArtists()
        
        self.artists.robots.omni = self.ax.scatter([], [], s = 25, c = 'C1', marker = 'o')
        # We create one object for the unicycles because each of them needs to
        # have a different dimension.
        # self.artists.robots.unicycles = [None] * self.n_robots[1]
        # for i in range(self.n_robots[1]):
        #     self.artists.robots.unicycles[i] = self.ax.scatter([], [], 25, 'C0')
        
        if self.artist_flags.centroid:
            self.artists.centroid = self.ax.scatter([], [], 25, 'C2')
        
        if self.artist_flags.voronoi:
            self.artists.voronoi = [self.ax.plot([],[])]
        
        if self.artist_flags.past_trajectory:
            self.artists.past_trajectory = [self.ax.plot([],[])[0] for _ in range(sum(self.n_robots)+1)]
        
        if self.artist_flags.goals and self.goals is not None:
            self.artists.goals = {'pos': None, 'name': None}
            self.artists.goals['pos'] = self.ax.scatter(
                [g[0] for g in self.goals] if self.goals is not None else [],
                [g[1] for g in self.goals] if self.goals is not None else [],
                25, 'k', 'x'
            )
        
            self.artists.goals['name'] = []
            if self.goals is not None:
                for i, g_i in enumerate(self.goals):
                    self.artists.goals['name'].append(
                        self.ax.annotate(
                            "$\mathcal{T}_{" + str(i+1) + "}$",
                            (g_i[0], g_i[1]+1),
                        )
                    )
        
        if self.artist_flags.obstacles and self.obstacles is not None:
            self.artists.obstacles = plt.Circle(self.obstacles[0:2], self.obstacles[2], color='grey', alpha=0.5)
            self.ax.add_patch(self.artists.obstacles)
            
        # ========================= Extract The State ======================== #
            
        # state = self.data[0]
        
        # x = [
        #     np.zeros((self.n_robots[0], 2)),
        #     np.zeros((self.n_robots[1], 3)),
        # ]
        
        # for c, state_c in enumerate(state):
        #     for j, s_c_j in enumerate(state_c):
        #         x[c][j, 0] = s_c_j[0]
        #         x[c][j, 1] = s_c_j[1]
        #         if c == 1:
        #             x[c][j, 2] = s_c_j[2]
        
        # # Unicycles.
        # for i in range(self.n_robots[1]):
        #     deg = x[0][i,2] * 180 / np.pi
        #     marker, scale = gen_arrow_head_marker(deg)
            
        #     plt.scatter(
        #         x = x[1][i,0], y = x[1][i,1],
        #         s = 250 * scale**2, c = 'C0',
        #         alpha=0.25,
        #         marker = marker,
        #     )
        
        # ============================== Legend ============================== #
        
        marker, scale = gen_arrow_head_marker(0)
        legend_elements = []
        if self.n_robots[0] > 0:
            legend_elements.append(
                Line2D([], [], marker='o', color='C1', linestyle='None', label='Omnidirectional robot')
            )
        # if self.n_robots[1] > 0:
        #     legend_elements.append(
        #         Line2D([], [], marker=marker, markersize=20*scale, color='C0', linestyle='None', label='Unicycle')
        #     )
        if self.artist_flags.centroid:
            legend_elements.append(
                Line2D([], [], marker='o', color='C2', linestyle='None', label='Fleet centroid')
            )
        if self.artist_flags.goals and self.goals is not None:
            if len(self.goals) > 0:
                legend_elements.append(
                    Line2D([], [], marker='x', color= 'k', linestyle='None', label='Goal')
                )
        if self.artist_flags.obstacles and self.obstacles is not None:
            legend_elements.append(
                plt.Circle([0,0], [0.1], color='grey', alpha=0.5, label='Obstacle')
            )
        
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        # =========================== Time On Plot =========================== #
        
        self.fr_number = self.ax.annotate(
            "$t = 0.00 \, s$",
            (0, 1),
            xycoords="axes fraction",
            xytext=(10, -10),
            fontsize=self.textsize,
            textcoords="offset points",
            ha="left",
            va="top",
        )
        
        # =================================================================== #
        
        self.ax.set_navigate_mode('pan')
    
    # ================================ Update ================================ #
    
    def update(self, frame):
        self.ax.figure.sca(self.ax)
        
        # ========================= Extract The State ======================== #
        
        # Current (frame) state.
        state = self.data[frame]
        
        x = [
            np.zeros((self.n_robots[0], 2)),
            # np.zeros((self.n_robots[1], 3)),
        ]
        
        for c, state_c in enumerate(state):
            for j, s_c_j in enumerate(state_c):
                x[c][j, 0] = s_c_j[0]
                x[c][j, 1] = s_c_j[1]
                if c == 1:
                    x[c][j, 2] = s_c_j[2]
                    
        # State history.
        n_history = min(self.n_history, frame)
        x_history = [
            np.zeros((self.n_robots[0], n_history, 2)),
            # np.zeros((self.n_robots[1], n_history, 3)),
        ]
        for k in range(n_history):
            for c in range(len(self.data[frame - k])):
                for j, s_c_j in enumerate(self.data[frame - k][c]):
                    x_history[c][j, k, 0] = s_c_j[0]
                    x_history[c][j, k, 1] = s_c_j[1]
                    if c == 1:
                        x_history[c][j, k, 2] = s_c_j[2]        
                        
        # ========================= Clean Old Artists ======================== #
        
        self.artists.robots.omni.remove()
        # for i in range(self.n_robots[1]):
        #     self.artists.robots.unicycle[i].remove()
        
        # ====================== Display Updated Artists ===================== #
        
        # Omnidirectional robot.
        self.artists.robots.omni = plt.scatter(
            x = x[0][:,0], y = x[0][:,1],
            s = 25, c = 'C1',
            marker = 'o',
        )
        
        # # Unicycles.
        # for i in range(self.n_robots[1]):
        #     deg = x[1][i,2] * 180 / np.pi
        #     marker, scale = gen_arrow_head_marker(deg)
                        
        #     self.artists.robots.unicycle[i] = plt.scatter(
        #         x = x[1][i,0], y = x[1][i,1],
        #         s = 250 * scale**2, c = 'C0',
        #         marker = marker,
        #     )
        
        # Fleet centroid. Plotted only if more than one robot.
        if self.artist_flags.centroid and sum(self.n_robots) > 1:
            self.artists.centroid.set_offsets(
                sum([np.nan_to_num(np.mean(
                    x[i][:,0:2],axis=0))*self.n_robots[i] for i in range(len(self.n_robots))]
                ) / sum(self.n_robots)
            )
        
        # Voronoi.
        if self.artist_flags.voronoi:
            towers = np.array(
                [e[0:2] for e in state[0]] #+ [e[0:2] for e in state[1]]
            )
            bounding_box = np.array([-20, 20, -20, 20])
            vor = BoundedVoronoi(towers, bounding_box)
            for v in self.artists.voronoi:
                try:
                    v.pop(0).remove()
                except:
                    v.remove()
            self.artists.voronoi = vor.plot()
        
        # Past trajectory.
        if self.artist_flags.past_trajectory:
            for e in self.artists.past_trajectory:
                e.remove()
            
            cnt = 0
            for c in range(len(self.data[frame])):
                for j, s_c_j in enumerate(state[c]):
                    self.artists.past_trajectory[cnt] = plt.plot(
                        x_history[c][j,:,0], x_history[c][j,:,1],
                        color = 'k',
                        linestyle = '--',
                        alpha = 0.5,
                    )[0]
                    cnt += 1
                    
            # Sum of x_history along c and j indices
            x_centroid_hist = np.sum(x_history[0], axis=(0)) / self.n_robots[0]
            self.artists.past_trajectory[cnt] = plt.plot(
                x_centroid_hist[:,0], x_centroid_hist[:,1],
                color = 'C2',
                linestyle = '--',
                alpha = 0.75,
            )[0]
        
        # Time on plot.
        self.fr_number.set_text(f"$t = {frame*self.dt:.2f} \, s$")
        
        return self.artists

# ============================= Display_animation ============================ #

def display_animation(
    s_history, goals, obstacles,
    dt: float, method: str = 'plot',
    artist_flags = MultiRobotArtists,
    x_lim = [-20., 20.], y_lim = [-20., 20.],
):
    fig, ax = plt.subplots()
    
    anim = Animation(s_history, artist_flags, goals, obstacles, ax, dt)
    anim.x_lim = x_lim
    anim.y_lim = y_lim
    
    n_steps = len(s_history)
    
    if method == 'plot':    
        ani = Player(
            fig=fig, func=anim.update, init_func=anim.init, frames=range(n_steps),
            maxi=n_steps-1, interval=dt*1000
        )
    elif method == 'save':
        ani = FuncAnimation(
            fig=fig, func=anim.update, init_func=anim.init, frames=range(n_steps),
            interval=dt*1000
        )
    
    if method == 'plot':
        plt.show()
    elif method == 'save':
        writervideo = FFMpegWriter(fps=int(1 / dt))
        ani.save('video.mp4', writer=writervideo)
    else:
        raise ValueError('The input method is {method}. Acceptable values are ' +
                         'plot, save, and none.')
    
# ============================== Save_snapshots ============================== #

def save_snapshots(
    s_history, goals, obstacles,
    dt: float, times: int | list[int], filename: str,
    artist_flags = MultiRobotArtists,
    x_lim = [-20., 20.], y_lim = [-20., 20.],
):
    if isinstance(times, int):
        times = [times]
    
    for time in times:
        frame = int(time / dt)
        
        fig, ax = plt.subplots()
        anim = Animation(s_history, artist_flags, goals, obstacles, ax, dt)
        anim.x_lim = x_lim
        anim.y_lim = y_lim
    
        anim.init()
        anim.update(frame)
        
        plt.savefig(f"{filename}_{time}.pdf", bbox_inches='tight', format='pdf')
        
        plt.close()
        
# ============================== Plot_distances ============================== #

def plot_distances(s_history, dt: float):
    [x_size_def, y_size_def] = plt.rcParams.get('figure.figsize')
    
    n_k = len(s_history)
    n_c = len(s_history[0])
    n_j = [len(s_history[0][c]) for c in range(n_c)]
    n_coord = 2
    
    x_hist = np.zeros([n_k, sum(n_j), n_coord])
    
    for k, c in np.ndindex(n_k, n_c):
        for j in range(n_j[c]):
            x_hist[k, sum(n_j[:c]) + j] = s_history[k][c][j][:n_coord]
            
    fig = plt.figure(figsize=(x_size_def, y_size_def/2))
    ax = plt.gca()
    
    for i, j in itertools.combinations(range(sum(n_j)), 2):
        ax.plot(
            np.arange(0, n_k*dt, dt),
            np.maximum(np.linalg.norm(x_hist[:,i] - x_hist[:,j], axis=1), 0.1),
        )
        
    ax.set(xlim=[0., 20.], ylim=[0., 4.5])
    ax.set_xlabel('Time [$s$]')
    ax.set_ylabel('Inter-robot dist. [$m$]')
    
    plt.savefig("distances.pdf", bbox_inches='tight', format='pdf')
