from dataclasses import dataclass

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

# ================================= Animation ================================ #

class Animation():
    def __init__(self, data, ax, dt) -> None:
        self.n_history = 50
        
        self.data = data
        self.ax = ax
        self.dt = dt
        
        self.n_robots = [len(data_i) for data_i in data[0]]
        
        self.artists = None
    
    # ================================= Init ================================= #
        
    def init(self):
        self.ax.clear()
        
        self.ax.set_aspect('equal', 'box')
        
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
        
        self.ax.set(xlim=[-20., 20.], ylim=[-20., 20.], xlabel='x [m]', ylabel='y [m]')
        
        marker, scale = gen_arrow_head_marker(0)
        legend_elements = [
            Line2D([], [], marker=marker, markersize=20*scale, color='C0', linestyle='None', label='Unicycles'),
            Line2D([], [], marker='o', color='C1', linestyle='None', label='Omnidirectional Robot'),
            Line2D([], [], marker='o', color='C2', linestyle='None', label='Fleet Centroid'),
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        self.fr_number = self.ax.annotate(
            "0",
            (0, 1),
            xycoords="axes fraction",
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            va="top",
        )
        
        # self.ax.axis('equal')
    
    # ================================ Update ================================ #
    
    def update(self, frame):
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
                
        for i in range(self.n_robots[0]):
            self.artists.unicycles[i].remove()

        self.artists.omnidir.remove()
                
        for i in range(self.n_robots[0]):
            deg = x[0][i,2] * 180 / np.pi
            marker, scale = gen_arrow_head_marker(deg)
                        
            self.artists.unicycles[i] = plt.scatter(
                x = x[0][i,0], y = x[0][i,1],
                s = 250 * scale**2, c = 'C0',
                marker = marker,
            )
            
        self.artists.omnidir = plt.scatter(
            x = x[1][:,0], y = x[1][:,1],
            s = 25, c = 'C1',
            marker = 'o',
        )
                
        self.artists.centroid.set_offsets(
            sum([np.nan_to_num(np.mean(x[i][:,0:2],axis=0))*self.n_robots[i] for i in range(len(self.n_robots))]) / sum(self.n_robots)
        )
         
        towers = np.array(
            [e[0:2] for e in state[0]] + 
            [e[0:2] for e in state[1]]
        )
        bounding_box = np.array([-20, 20, -20, 20])
        vor = BoundedVoronoi(towers, bounding_box)
        for v in self.artists.voronoi:
            v.pop(0).remove()
        self.artists.voronoi = vor.plot()
        
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
        
        self.fr_number.set_text(f"t: {frame*self.dt:.2f} s")
        
        return self.artists

# ============================= Display_animation ============================ #

def display_animation(s_history, dt, method = 'plot'):
    fig, ax = plt.subplots()
    
    anim = Animation(s_history, ax, dt)
    
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
        
def save_snapshots(s_history, dt, times: int | list[int], filename):
    fig, ax = plt.subplots()
    
    anim = Animation(s_history, ax, dt)
    
    if isinstance(times, int):
        times = [times]
    
    for time in times:
        frame = int(time / dt)
    
        anim.init()
        anim.update(frame)
        
        plt.savefig(f"{filename}_{time}.svg", bbox_inches='tight', format='svg')
        
        plt.close()
