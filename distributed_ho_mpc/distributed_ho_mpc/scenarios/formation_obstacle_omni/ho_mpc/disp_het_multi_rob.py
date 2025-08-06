import itertools
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.widgets
import mpl_toolkits.axes_grid1
import numpy as np
from cycler import cycler
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D

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

    arr = np.array([[0.1, 0.3], [0.1, -0.3], [1, 0], [0.1, 0.3]])  # arrow shape
    angle = rot / 180 * np.pi
    rot_mat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))
    codes = [
        mpl.path.Path.MOVETO,
        mpl.path.Path.LINETO,
        mpl.path.Path.LINETO,
        mpl.path.Path.CLOSEPOLY,
    ]
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
    obstacles: ...


# =========================================================================== #


class Player(FuncAnimation):
    def __init__(
        self,
        fig,
        func,
        frames=None,
        init_func=None,
        fargs=None,
        save_count=None,
        mini=0,
        maxi=100,
        pos=(0.4, 0.15),
        **kwargs,
    ):
        self.i = 0
        self.min = mini
        self.max = maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(
            self,
            self.fig,
            self.func,
            frames=self.play(),
            init_func=init_func,
            fargs=fargs,
            save_count=save_count,
            **kwargs,
        )

    def play(self):
        while self.runs:
            self.i = self.i + self.forwards - (not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs = True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i + self.forwards - (not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.func(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        button_font = {'family': 'sans-serif', 'size': 12}

        playerax = self.fig.add_axes([pos[0], pos[1], 0.22, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes('right', size='80%', pad=0.05)
        sax = divider.append_axes('right', size='80%', pad=0.05)
        fax = divider.append_axes('right', size='80%', pad=0.05)
        ofax = divider.append_axes('right', size='100%', pad=0.05)
        self.button_oneback = matplotlib.widgets.Button(playerax, label=r'\faStepBackward')
        self.button_back = matplotlib.widgets.Button(bax, label=r'\faStepBackward')
        self.button_stop = matplotlib.widgets.Button(sax, label=r'\faPause')
        self.button_forward = matplotlib.widgets.Button(fax, label=r'\faPlay')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label=r'\faStepForward')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)


def init_matplotlib():
    default_cycler = cycler(
        color=[
            '#0072BD',
            '#D95319',
            '#EDB120',
            '#7E2F8E',
            '#77AC30',
            '#4DBEEE',
            '#A2142F',
            '#FF6F00',
            '#8DFF33',
            '#33FFF7',
        ]
    ) + cycler('linestyle', ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--'])

    textsize = 16
    labelsize = 18

    plt.rc('font', family='serif', serif='Times')
    plt.rcParams['text.usetex'] = True
    plt.rc(
        'text.latex',
        preamble=r'\usepackage[utf8]{inputenc} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{fontawesome} \DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}',
    )
    plt.rc('xtick', labelsize=textsize)
    plt.rc('ytick', labelsize=textsize)
    plt.rc('axes', titlesize=labelsize, labelsize=labelsize, prop_cycle=default_cycler)
    plt.rc('legend', fontsize=textsize)
    plt.rc('grid', linestyle='-.', alpha=0.5)
    plt.rc('axes', grid=True)

    plt.rcParams['figure.constrained_layout.use'] = True

    return textsize


# ================================= Animation ================================ #


class Animation:
    def __init__(self, data, goals, obstacles, ax, dt, estim) -> None:
        self.textsize = init_matplotlib()

        self.n_history = np.inf

        self.data = data
        self.goals = goals
        self.obstacles = obstacles
        self.ax = ax
        self.dt = dt

        self.x_lim = [-20.0, 20.0]
        self.y_lim = [-20.0, 20.0]

        self.show_estimation = estim
        self.n_robots = [len(data_i) for data_i in data[0]]

        # positional reference of neighbour estimation inside the data
        self.pos_neigh = [None for i in range(len(self.n_robots))]
        for p, l in enumerate(self.n_robots):
            if p == 0:
                self.pos_neigh[p] = 0
            else:
                self.pos_neigh[p] = sum(self.n_robots[:p]) - 1 * p

        self.artists = None

        self.show_trajectory = True
        self.show_voronoi = True

    # ================================= Init ================================= #

    def init(self):
        self.ax.clear()

        self.ax.set_aspect('equal', 'box')

        # ====================== Initialize The Artists ====================== #

        self.artists = MultiRobotArtists
        self.artists.unicycles = [None for _ in self.n_robots]
        self.artists.omnidir = [None for _ in self.n_robots]

        # We create one object for the unicycles because each of them needs to
        # have a different dimension.
        for i in range(len(self.n_robots)):
            self.artists.unicycles[i] = self.ax.scatter([], [], 25, 'C0')

            self.artists.omnidir[i] = self.ax.scatter(
                [],
                [],
                s=25,
                c='C1',
                marker='o',
            )

        self.artists.centroid = self.ax.scatter([], [], 25, 'C2')

        self.artists.voronoi = [self.ax.plot([], [])]

        if self.show_trajectory:
            self.artists.past_trajectory = [
                self.ax.plot([], []) for _ in range(len(self.n_robots))
            ]  # sum(self.n_robots)+1
            self.artists.past_trajectory = [e[0] for e in self.artists.past_trajectory]

        self.ax.set(xlim=self.x_lim, ylim=self.y_lim, xlabel='$x$ [$m$]', ylabel='$y$ [$m$]')

        self.artists.goals = [None for _ in range(2)]
        self.artists.goals[0] = self.ax.scatter(
            [g[0] for g in self.goals] if self.goals is not None else [],
            [g[1] for g in self.goals] if self.goals is not None else [],
            25,
            'k',
            'x',
        )

        self.artists.goals[1] = []
        if self.goals is not None:
            for i, g_i in enumerate(self.goals):
                self.artists.goals[1].append(
                    self.ax.annotate(
                        '$\mathcal{T}_{' + str(i + 1) + '}$',
                        (g_i[0], g_i[1] + 1),
                    )
                )

        # state = self.data[0]

        # x = [
        #     np.zeros((self.n_robots[0], 3)),
        #     np.zeros((self.n_robots[1], 2)),
        # ]

        # for c, state_c in enumerate(state):
        #     for j, s_c_j in enumerate(state_c):
        #         x[c][j, 0] = s_c_j[0]
        #         x[c][j, 1] = s_c_j[1]
        #         if c == 0:
        #             x[c][j, 2] = s_c_j[2]

        # # Unicycles.
        # for i in range(self.n_robots[0]):
        #     deg = x[0][i,2] * 180 / np.pi
        #     marker, scale = gen_arrow_head_marker(deg)

        #     plt.scatter(
        #         x = x[0][i,0], y = x[0][i,1],
        #         s = 250 * scale**2, c = 'C0',
        #         alpha=0.25,
        #         marker = marker,
        #     )

        if self.obstacles is not None:
            self.artists.obstacles = plt.Circle(
                self.obstacles[0:2], self.obstacles[2], color='grey', alpha=0.5
            )
            self.ax.add_patch(self.artists.obstacles)

        # ============================== Legend ============================== #

        marker, scale = gen_arrow_head_marker(0)
        legend_elements = []
        if self.n_robots[0] > 0:
            for i in range(len(self.n_robots)):
                legend_elements.append(
                    Line2D(
                        [],
                        [],
                        marker=marker,
                        markersize=20 * scale,
                        color=f'C{i}',
                        linestyle='None',
                        label=f'Agent_{i}',
                    )
                )
        """if self.n_robots[1] > 0:
            legend_elements.append(
                Line2D([], [], marker='o', color=f'C{i}', linestyle='None', label='Neigh_copy')
            )"""
        """if sum(self.n_robots) > 1:
            legend_elements.append(
                Line2D([], [], marker='o', color='C2', linestyle='None', label='Fleet centroid')
            )"""
        if self.goals is not None:
            if len(self.goals) > 0:
                legend_elements.append(
                    Line2D([], [], marker='x', color='k', linestyle='None', label='Goal')
                )
        if self.obstacles is not None:
            legend_elements.append(
                plt.Circle([0, 0], [0.1], color='grey', alpha=0.5, label='Obstacle')
            )

        # self.ax.legend(handles=legend_elements, loc='upper right')

        # =========================== Time On Plot =========================== #

        self.fr_number = self.ax.annotate(
            '$t = 0.00 \, s$',
            (0, 1),
            xycoords='axes fraction',
            xytext=(10, -10),
            fontsize=self.textsize,
            textcoords='offset points',
            ha='left',
            va='top',
        )

        # =================================================================== #

        self.ax.set_navigate_mode('pan')

    # ================================ Update ================================ #

    def update(self, frame):
        self.ax.figure.sca(self.ax)

        # ========================= Extract The State ======================== #

        # Current (frame) state.
        state = self.data[frame]
        # x = [np.zeros((agent, 2)) for agent in self.n_robots]

        n_j = 0 - len(self.n_robots)
        for n in self.n_robots:
            n_j += n

        x = [
            np.zeros((len(self.n_robots), 2)),
            np.zeros((n_j, 2)),
        ]

        p = 0
        for i, state_c in enumerate(state):
            for j, s_c_j in enumerate(state_c):
                if j == 0:
                    x[0][i, 0] = s_c_j[0]
                    x[0][i, 1] = s_c_j[1]
                else:
                    x[1][p, 0] = s_c_j[0]
                    x[1][p, 1] = s_c_j[1]
                    p += 1

        # State history.
        n_history = min(self.n_history, frame)
        # x_history = [
        #    np.zeros((len(self.n_robots), n_history, 3)),
        #    np.zeros((n_j, n_history, 2)),
        # ]
        x_history = [np.zeros((agent, n_history, 2)) for agent in self.n_robots]
        p = 0
        for k in range(n_history):
            for c in range(len(self.data[frame - k])):
                for j, s_c_j in enumerate(self.data[frame - k][c]):
                    x_history[c][j, k, 0] = s_c_j[0]
                    x_history[c][j, k, 1] = s_c_j[1]
                    # if c == 0:
                    #     x_history[c][j, k, 2] = s_c_j[2]

        # ========================= Clean Old Artists ======================== #

        for i in range(len(self.n_robots)):
            self.artists.unicycles[i].remove()

            if not self.show_estimation:
                continue
            self.artists.omnidir[i].remove()

        # ====================== Display Updated Artists ===================== #

        # Unicycles.
        for i in range(len(self.n_robots)):
            # deg = x[0][i,2] * 180 / np.pi
            deg = 1
            marker, scale = gen_arrow_head_marker(deg)

            self.artists.unicycles[i] = plt.scatter(
                x=x[0][i, 0],
                y=x[0][i, 1],
                s=40,
                c=f'C{i}',
                marker='o',
            )

            if not self.show_estimation:
                continue
            # Omnidirectional robot.
            self.artists.omnidir[i] = plt.scatter(
                x=x[1][self.pos_neigh[i] : (self.pos_neigh[i] + self.n_robots[i] - 1), 0],
                y=x[1][self.pos_neigh[i] : (self.pos_neigh[i] + self.n_robots[i] - 1), 1],
                s=25,
                c=f'C{i}',
                marker='o',
            )

        """# TODO: handle myself robots (style) in plot
        # real robot.
        for i in range(len(self.n_robots)):
            #deg = x[0][i,2] * 180 / np.pi
            deg = 1
            marker, scale = gen_arrow_head_marker(deg)
                        
            self.artists.unicycles[i] = plt.scatter(
                x = x[i][0,0], y = x[i][0,1],
                s = 250 * scale**2, c = 'C0',
                marker = marker,
            )"""

        """#Omnidirectional robot.
        self.artists.omnidir = plt.scatter(
            x = x[1][:,0], y = x[1][:,1],
            s = 25, c = 'C1',
            marker = 'o',
        )"""

        """# TODO: handle neigh robots (style) in plot
        # neighbours robot.
        for i in range(len(self.n_robots)):
            self.artists.omnidir = plt.scatter(
                x = x[i][1:,0], y = x[i][1:,1],
                s = 25, c = 'C1',
                marker = 'o',
            )"""

        # Fleet centroid. Plotted only if more than one robot.
        # if sum(self.n_robots) > 1:
        #     self.artists.centroid.set_offsets(
        #         sum([np.nan_to_num(np.mean(
        #             x[i][:,0:2],axis=0))*self.n_robots[i] for i in range(len(self.n_robots))]
        #         ) / sum(self.n_robots)
        #     )

        # Voronoi.
        if self.show_voronoi:
            towers = np.array([e[0:2] for e in state[0]] + [e[0:2] for e in state[1]])
            bounding_box = np.array([-20, 20, -20, 20])
            vor = BoundedVoronoi(towers, bounding_box)
            for v in self.artists.voronoi:
                try:
                    v.pop(0).remove()
                except:
                    v.remove()
            self.artists.voronoi = vor.plot()

        # Past trajectory.
        if self.show_trajectory:
            for e in self.artists.past_trajectory:
                e.remove()

            cnt = 0
            for c in range(len(self.data[frame])):
                for j, s_c_j in enumerate(state[c]):
                    if j > 0:
                        continue
                    self.artists.past_trajectory[cnt] = plt.plot(
                        x_history[c][j, :, 0],
                        x_history[c][j, :, 1],
                        color='k',
                        linestyle='--',
                        alpha=0.5,
                    )[0]
                    cnt += 1

            ## Sum of x_history along c and j indices
            # x_centroid_hist = np.sum(x_history[1], axis=(0)) / self.n_robots[1]
            # self.artists.past_trajectory[cnt] = plt.plot(
            #    x_centroid_hist[:,0], x_centroid_hist[:,1],
            #    color = 'C2',
            #    linestyle = '--',
            #    alpha = 0.75,
            # )[0]

        # Time on plot.
        self.fr_number.set_text(f'$t = {frame * self.dt:.2f} \, s$')

        return self.artists


# ============================= Display_animation ============================ #


def display_animation(
    s_history,
    goals,
    obstacles,
    dt: float,
    method: str = 'plot',
    show_trajectory: bool = True,
    show_voronoi: bool = True,
    estim: bool = True,
    x_lim=[-20.0, 25.0],
    y_lim=[-20.0, 25.0],
):
    fig, ax = plt.subplots()

    anim = Animation(s_history, goals, obstacles, ax, dt, estim)
    anim.show_trajectory = show_trajectory
    anim.show_voronoi = show_voronoi
    anim.x_lim = x_lim
    anim.y_lim = y_lim

    n_steps = len(s_history)

    if method == 'plot':
        ani = Player(
            fig=fig,
            func=anim.update,
            init_func=anim.init,
            frames=range(n_steps),
            maxi=n_steps - 1,
            interval=dt * 1000,
        )
    elif method == 'save':
        ani = FuncAnimation(
            fig=fig,
            func=anim.update,
            init_func=anim.init,
            frames=range(n_steps),
            interval=dt * 1000,
        )

    if method == 'plot':
        plt.show()
    elif method == 'save':
        writervideo = FFMpegWriter(fps=int(1 / dt))
        ani.save('video.mp4', writer=writervideo)
    else:
        raise ValueError(
            'The input method is {method}. Acceptable values are ' + 'plot, save, and none.'
        )


# ============================== Save_snapshots ============================== #


def save_snapshots(
    s_history,
    goals,
    obstacles,
    dt: float,
    times: int | list[int],
    filename: str,
    show_trajectory: bool = True,
    show_voronoi: bool = True,
    estim: bool = False,
    x_lim=[-7.0, 20.0],
    y_lim=[-7.0, 20.0],
):
    if isinstance(times, int):
        times = [times]

    for time in times:
        frame = int(time / dt)

        fig, ax = plt.subplots()
        anim = Animation(s_history, goals, obstacles, ax, dt, estim)
        anim.show_trajectory = show_trajectory
        anim.show_voronoi = show_voronoi
        anim.x_lim = x_lim
        anim.y_lim = y_lim

        anim.init()
        anim.update(frame)

        plt.savefig(f'{filename}_{time}.pdf', bbox_inches='tight', format='pdf')

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

    fig = plt.figure(figsize=(x_size_def, y_size_def / 2))
    ax = plt.gca()

    for i, j in itertools.combinations(range(sum(n_j)), 2):
        ax.plot(
            np.arange(0, n_k * dt, dt),
            np.maximum(np.linalg.norm(x_hist[:, i] - x_hist[:, j], axis=1), 0.1),
        )

    ax.set(xlim=[0.0, 20.0], ylim=[0.0, 4.5])
    ax.set_xlabel('Time [$s$]')
    ax.set_ylabel('Inter-robot dist. [$m$]')

    plt.savefig('distances.pdf', bbox_inches='tight', format='pdf')
