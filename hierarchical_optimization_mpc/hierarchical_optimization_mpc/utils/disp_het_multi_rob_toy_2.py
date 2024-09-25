from dataclasses import dataclass

from cycler import cycler
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from hierarchical_optimization_mpc.voronoi_task import BoundedVoronoi



def gen_arrow_head_marker(rot):
    """generate a marker to plot with matplotlib scatter, plot, ...

    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    rot=0: positive x direction
    Parameters
    ----------
    rot : float
        rotation in degree
        0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """
    
    arr = np.array([[.1, .3], [.1, -.3], [1, 0], [.1, .3]])  # arrow shape
    angle = rot / 180 * np.pi
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
        ])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))
    codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO,mpl.path.Path.LINETO, mpl.path.Path.CLOSEPOLY]
    arrow_head_marker = mpl.path.Path(arr, codes)
    
    return arrow_head_marker, scale

@dataclass
class MultiRobotArtists:
    unicycles: ...
    omnidir: ...
    centroid: ...
    voronoi: ...
    past_trajectory: ...
    goals: ...

# ================================= Animation ================================ #

class Animation():
    default_cycler = (
        cycler(color=['#0072BD', '#D95319', '#EDB120', '#7E2F8E']) +
        cycler('linestyle', ['-', '--', '-', '--'])
    )
    
    textsize = 16
    labelsize = 18
    
    plt.rc('font', family='serif', serif='Times')
    plt.rcParams["text.usetex"] = True
    plt.rc('xtick', labelsize=textsize)
    plt.rc('ytick', labelsize=textsize)
    plt.rc('axes', titlesize=labelsize, labelsize=labelsize, prop_cycle=default_cycler)
    plt.rc('legend', fontsize=textsize)
    plt.rc('grid', linestyle='-.', alpha=0.5)
    plt.rc('axes', grid=True)
    
    plt.rcParams['figure.constrained_layout.use'] = True
    
    def __init__(self, data, goals, ax, dt) -> None:
        self.n_history = np.inf
        
        self.data = data[0]
        self.data1 = data[1]
        self.data2 = data[2]
        self.goals = goals
        self.ax = ax
        self.dt = dt
        
        self.n_robots = [len(data_i) for data_i in data[0][0]]
        
        self.artists = None
    
    # ================================= Init ================================= #
        
    def init(self):
        self.ax.clear()
        
        self.ax.set_aspect('equal', 'box')
        
        # ====================== Initialize The Artists ====================== #
        
        self.artists = MultiRobotArtists
        self.artists.unicycles = [None] * self.n_robots[0]
        self.artists.omnidir = [None] * self.n_robots[1]
        
        # We create one object for the unicycles because each of them needs to
        # have a different dimension.    
        for i in range(self.n_robots[0]):
            self.artists.unicycles[i] = self.ax.scatter([], [], 25, 'C0')
        
        self.artists.omnidir = self.ax.scatter([], [], s = 25, c = 'C1', marker = 'o',)
            
        self.artists.centroid = self.ax.scatter([], [], 25, 'C2')
        
        self.artists.voronoi = [self.ax.plot([],[])]
        
        self.artists.past_trajectory = [self.ax.plot([],[]) for _ in range(sum(self.n_robots))]
        self.artists.past_trajectory = [e[0] for e in self.artists.past_trajectory]
        
        self.ax.set(xlim=[-20., 20.], ylim=[-20., 20.], xlabel='$x$ [$m$]', ylabel='$y$ [$m$]')
        
        self.artists.goals = self.ax.scatter(
            [g[0] for g in self.goals] if self.goals is not None else [],
            [g[1] for g in self.goals] if self.goals is not None else [],
            25, 'k', 'x'
        )
        
        # ============================== Legend ============================== #
        
        marker, scale = gen_arrow_head_marker(0)
        legend_elements = []
        if self.n_robots[0] > 0:
            legend_elements.append(
                Line2D([], [], marker=marker, markersize=20*scale, color='C0', linestyle='None', label='Unicycle')
            )
        if self.n_robots[1] > 0:
            legend_elements.append(
                Line2D([], [], marker='o', color='C1', linestyle='None', label='Omnidirectional robot')
            )
        if sum(self.n_robots) > 1:
            legend_elements.append(
                Line2D([], [], marker='o', color='C2', linestyle='None', label='Fleet centroid')
            )
        if self.goals is not None:
            if len(self.goals) > 0:
                legend_elements.append(
                    Line2D([], [], marker='x', color= 'k', linestyle='None', label='Goal')
                )
        legend_elements.append(
            Line2D([0,1], [0,0], color= 'k', linestyle='--', alpha = 0.5, label='Trajectory -- prioritized')
        )
        legend_elements.append(
            Line2D([0,1], [0,0], color= 'b', linestyle='--', alpha = 0.5, label='Trajectory -- weighted $\kappa=5$')
        )
        legend_elements.append(
            Line2D([0,1], [0,0], color= 'r', linestyle='--', alpha = 0.5, label='Trajectory -- weighted $\kappa=100$')
        )
        
        self.ax.legend(handles=legend_elements, loc='lower left')
        
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
    
    # ================================ Update ================================ #
    
    def update(self, frame):
        
        # ========================= Extract The State ======================== #
        
        # Current (frame) state.
        state = self.data[frame]
        
        x = [
            np.zeros((self.n_robots[0], 3)),
            np.zeros((self.n_robots[1], 2)),
        ]
        
        for c, state_c in enumerate(state):
            for j, s_c_j in enumerate(state_c):
                x[c][j, 0] = s_c_j[0]
                x[c][j, 1] = s_c_j[1]
                if c == 0:
                    x[c][j, 2] = s_c_j[2]
                    
        # State history.
        n_history = min(self.n_history, frame)
        x_history = [
            np.zeros((self.n_robots[0], n_history, 3)),
            np.zeros((self.n_robots[1], n_history, 2)),
        ]
        for k in range(n_history):
            for c in range(len(self.data[frame - k])):
                for j, s_c_j in enumerate(self.data[frame - k][c]):
                    x_history[c][j, k, 0] = s_c_j[0]
                    x_history[c][j, k, 1] = s_c_j[1]
                    if c == 0:
                        x_history[c][j, k, 2] = s_c_j[2]
                        
        x_history_1 = [
            np.zeros((self.n_robots[0], n_history, 3)),
            np.zeros((self.n_robots[1], n_history, 2)),
        ]
        for k in range(n_history):
            for c in range(len(self.data1[frame - k])):
                for j, s_c_j in enumerate(self.data1[frame - k][c]):
                    x_history_1[c][j, k, 0] = s_c_j[0]
                    x_history_1[c][j, k, 1] = s_c_j[1]
                    if c == 0:
                        x_history_1[c][j, k, 2] = s_c_j[2]
                        
        x_history_2 = [
            np.zeros((self.n_robots[0], n_history, 3)),
            np.zeros((self.n_robots[1], n_history, 2)),
        ]
        for k in range(n_history):
            for c in range(len(self.data2[frame - k])):
                for j, s_c_j in enumerate(self.data2[frame - k][c]):
                    x_history_2[c][j, k, 0] = s_c_j[0]
                    x_history_2[c][j, k, 1] = s_c_j[1]
                    if c == 0:
                        x_history_2[c][j, k, 2] = s_c_j[2]
                        
        # ========================= Clean Old Artists ======================== #
        
        for i in range(self.n_robots[0]):
            self.artists.unicycles[i].remove()

        self.artists.omnidir.remove()
        
        # ====================== Display Updated Artists ===================== #
        
        # Unicycles.
        for i in range(self.n_robots[0]):
            deg = x[0][i,2] * 180 / np.pi
            marker, scale = gen_arrow_head_marker(deg)
                        
            self.artists.unicycles[i] = plt.scatter(
                x = x[0][i,0], y = x[0][i,1],
                s = 250 * scale**2, c = 'C0',
                marker = marker,
            )
            
        # Omnidirectional robot.
        self.artists.omnidir = plt.scatter(
            x = x[1][:,0], y = x[1][:,1],
            s = 25, c = 'C1',
            marker = 'o',
        )
        
        # Fleet centroid. Plotted only if more than one robot.
        if sum(self.n_robots) > 1:
            self.artists.centroid.set_offsets(
                sum([np.nan_to_num(np.mean(
                    x[i][:,0:2],axis=0))*self.n_robots[i] for i in range(len(self.n_robots))]
                ) / sum(self.n_robots)
            )
        
        # Voronoi.
        towers = np.array(
            [e[0:2] for e in state[0]] + 
            [e[0:2] for e in state[1]]
        )
        bounding_box = np.array([-20, 20, -20, 20])
        vor = BoundedVoronoi(towers, bounding_box)
        for v in self.artists.voronoi:
            v.pop(0).remove()
        self.artists.voronoi = vor.plot()
        
        # Past trajectory.
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
                
        for c in range(len(self.data1[frame])):
            for j, s_c_j in enumerate(state[c]):
                plt.plot(
                    x_history_1[c][j,:,0], x_history_1[c][j,:,1],
                    color = 'b',
                    linestyle = '--',
                    alpha = 0.5,
                )
                
        for c in range(len(self.data2[frame])):
            for j, s_c_j in enumerate(state[c]):
                plt.plot(
                    x_history_2[c][j,:,0], x_history_2[c][j,:,1],
                    color = 'r',
                    linestyle = '--',
                    alpha = 0.5,
                )
        
        # Time on plot.
        self.fr_number.set_text(f"$t = {frame*self.dt:.2f} \, s$")
        
        return self.artists

# ============================= Display_animation ============================ #

def display_animation(s_history, goals, dt: float, method: str = 'plot'):
    fig, ax = plt.subplots()
    
    anim = Animation(s_history, goals, ax, dt)
    
    n_steps = len(s_history)
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

def save_snapshots(s_histories, goals, dt: float, times: int | list[int], filename: str):    
    if isinstance(times, int):
        times = [times]
    
    for time in times:
        frame = int(time / dt)
        
        fig, ax = plt.subplots()
        anim = Animation(s_histories, goals, ax, dt)
    
        anim.init()
        anim.update(frame)
        
        plt.savefig(f"{filename}_{time}.pdf", bbox_inches='tight', format='pdf')
        
        plt.close()
