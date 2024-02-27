from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np



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
class MultiRobotScatter:
    unicycles: ...
    omnidir: ...
    centroid: ...

class Animation():
    def __init__(self, scat: MultiRobotScatter, data) -> None:
        self.scat = scat
        self.data = data
        
        self.n_robots = [len(data_i) for data_i in data[0]]
    
    def update(self, frame):
        state = self.data[frame]
        
        x = [
            np.zeros((self.n_robots[0], 3)),
            np.zeros((self.n_robots[1], 2)),
        ]
        
        for c in range(len(state)):
            for j, s_c_j in enumerate(state[c]):
                x[c][j, 0] = s_c_j[0]
                x[c][j, 1] = s_c_j[1]
                if c == 0:
                    x[c][j, 2] = s_c_j[2]
                
        for i in range(self.n_robots[0]):
            self.scat.unicycles[i].remove()
        for i in range(self.n_robots[1]):
            self.scat.omnidir[i].remove()
                
        for i in range(self.n_robots[0]):
            deg = x[0][i,2] * 180 / np.pi
            marker, scale = gen_arrow_head_marker(deg)
                        
            self.scat.unicycles[i] = plt.scatter(
                x = x[0][i,0], y = x[0][i,1],
                s = 250 * scale**2, c = 'C0',
                marker = marker,
            )
            
        for i in range(self.n_robots[1]):
            self.scat.omnidir[i] = plt.scatter(
                x = x[1][i,0], y = x[1][i,1],
                s = 25, c = 'C1',
                marker = 'o',
            )
                
        self.scat.centroid.set_offsets(
            sum([np.nan_to_num(np.mean(x[i][:,0:2],axis=0))*self.n_robots[i] for i in range(len(self.n_robots))]) / sum(self.n_robots)
        )
        
        return self.scat