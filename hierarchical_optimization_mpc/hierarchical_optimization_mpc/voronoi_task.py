import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import sys

class BoundedVoronoi:
    def __init__(self, towers, bounding_box):
        # Select towers inside the bounding box
        i = self.in_box(towers, bounding_box)
        # Mirror points
        points_center = towers[i, :]
        
        points_left = np.copy(points_center)
        points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
        points_right = np.copy(points_center)
        points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
        points_down = np.copy(points_center)
        points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
        points_up = np.copy(points_center)
        points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])

        points = np.concatenate((points_center, points_left, points_right, points_up, points_down))
        
        # Compute Voronoi
        self.vor = spa.Voronoi(points)

        self.vor.filtered_points = points_center
        self.vor.filtered_regions = self.filter_regions(self.vor.regions, self.vor.vertices, bounding_box)
        self.vor.centroids = self.compute_centroids()
        
    @staticmethod
    def in_box(towers, bounding_box):
        return np.logical_and(
            np.logical_and(bounding_box[0] <= towers[:, 0], towers[:, 0] <= bounding_box[1]),
            np.logical_and(bounding_box[2] <= towers[:, 1], towers[:, 1] <= bounding_box[3])
        )
        
    @staticmethod
    def filter_regions(regions, vertices, bounding_box):
        eps = sys.float_info.epsilon * 1000
        filtered_regions = []
        for region in regions:
            flag = True
            for index in region:
                if index == -1:
                    flag = False
                    break
                x, y = vertices[index]
                if not (bounding_box[0] - eps <= x <= bounding_box[1] + eps and
                        bounding_box[2] - eps <= y <= bounding_box[3] + eps):
                    flag = False
                    break
            if region and flag:
                filtered_regions.append(region)
        return filtered_regions
        
    @staticmethod
    def region_centroid(vertices):
        A = 0.5 * np.sum(vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1])
        C_x = np.sum((vertices[:-1, 0] + vertices[1:, 0]) * (vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1]))
        C_y = np.sum((vertices[:-1, 1] + vertices[1:, 1]) * (vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1]))
        return np.array([[C_x / (6 * A), C_y / (6 * A)]])
    
    def compute_centroids(self):
        centroids = np.zeros((len(self.vor.filtered_regions), 2))
        for i, region in enumerate(self.vor.filtered_regions):
            centroids[i, :] = self.region_centroid(self.vor.vertices[region + [region[0]], :])
            
        return centroids
    
    def get_centroids_2(self):
        centroids = np.zeros((len(self.vor.filtered_regions), 2))
        point_region = self.vor.point_region[0:6]
        for i, idx in enumerate(point_region.tolist()):
            region = self.vor.regions[idx]
            centroids[i, :] = self.region_centroid(self.vor.vertices[region + [region[0]], :])
            
        return centroids
    
    def plot(self):
        # # Plot initial points
        # ax.plot(self.vor.filtered_points[:, 0], self.vor.filtered_points[:, 1], 'b.')

        # # Plot ridges points
        # for region in self.vor.filtered_regions:
        #     vertices = self.vor.vertices[region, :]
        #     ax.plot(vertices[:, 0], vertices[:, 1], 'go')
                
        vor_plot = ["" for _ in range(len(self.vor.filtered_regions))]
            
        # Plot ridges
        for i, region in enumerate(self.vor.filtered_regions):
            vertices = self.vor.vertices[region + [region[0]], :]
            vor_plot[i] = plt.plot(vertices[:, 0], vertices[:, 1], 'k-')
            
        # # Compute and plot centroids
        # ax.scatter(self.vor.centroids[:,0], self.vor.centroids[:,1])
            
        return vor_plot

# ============================================================================ #

def main():
    # np.random.seed(1)
    n_towers = 100
    towers = np.random.rand(n_towers, 2)
    bounding_box = np.array([0, 1, 0, 1]) # [x_min, x_max, y_min, y_max]

    vor = BoundedVoronoi(towers, bounding_box)
    
    print(vor.vor.centroids)

    fig = plt.figure()
    ax = fig.gca()

    vor.plot(ax)
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    
    plt.show()

if __name__ == '__main__':
    main()
