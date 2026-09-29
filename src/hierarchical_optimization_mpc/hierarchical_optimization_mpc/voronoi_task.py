import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi


class BoundedVoronoi(Voronoi):
    def __init__(self, towers, bounding_box, show_centroids=False):
        # Select towers inside the bounding box
        i = self.in_box(towers, bounding_box)

        points_center = towers[i, :]

        # Mirror points on the left, right, down, and up.
        points_left = np.copy(points_center)
        points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
        points_right = np.copy(points_center)
        points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
        points_down = np.copy(points_center)
        points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
        points_up = np.copy(points_center)
        points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])

        # Concatenate the points.
        points = np.concatenate((points_center, points_left, points_right, points_up, points_down))

        super().__init__(points)

        # Compute the Voronoi cells centroids
        self.centroids = self.compute_centroids()

        self.show_centroids = show_centroids

    @staticmethod
    def in_box(towers, bounding_box):
        return np.logical_and(
            np.logical_and(bounding_box[0] <= towers[:, 0], towers[:, 0] <= bounding_box[1]),
            np.logical_and(bounding_box[2] <= towers[:, 1], towers[:, 1] <= bounding_box[3]),
        )

    @staticmethod
    def region_centroid(vertices):
        A = 0.5 * np.sum(vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1])
        C_x = np.sum(
            (vertices[:-1, 0] + vertices[1:, 0])
            * (vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1])
        )
        C_y = np.sum(
            (vertices[:-1, 1] + vertices[1:, 1])
            * (vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1])
        )
        return np.array([[C_x / (6 * A), C_y / (6 * A)]])

    def compute_centroids(self):
        n_towers = self.npoints // 5
        centroids = np.zeros((n_towers, 2))

        for i, idx in enumerate(self.point_region[0:n_towers].tolist()):
            region = self.regions[idx]
            centroids[i, :] = self.region_centroid(self.vertices[region + [region[0]], :])

        return centroids

    def plot(self):
        n_towers = self.npoints // 5

        vor_plot = ['' for _ in range(n_towers)]

        # Plot ridges
        for i, idx in enumerate(self.point_region[0:n_towers].tolist()):
            region = self.regions[idx]
            vertices = self.vertices[region + [region[0]], :]
            vor_plot[i] = plt.plot(vertices[:, 0], vertices[:, 1], 'k-')

        if self.show_centroids:
            # Compute and plot centroids
            vor_plot.append(
                plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x')
            )

        return vor_plot


class VoronoiTask(BoundedVoronoi):
    def __init__(self, towers, bounding_box):
        super().__init__(towers, bounding_box)


# ============================================================================ #


def main():
    # np.random.seed(1)
    n_towers = 100
    towers = np.random.rand(n_towers, 2)
    bounding_box = np.array([0, 1, 0, 1])  # [x_min, x_max, y_min, y_max]

    b_vor = BoundedVoronoi(towers, bounding_box)

    fig = plt.figure()
    ax = fig.gca()

    b_vor.plot(ax)
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])

    plt.show()


if __name__ == '__main__':
    main()
