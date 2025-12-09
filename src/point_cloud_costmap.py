import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.filters import threshold_otsu
from skimage.feature import blob_log


def project_point_cloud_optimized(points, resolution=0.01):
    world_min_y = np.min(points[:, 1])
    world_max_y = np.max(points[:, 1])
    world_min_x = np.min(points[:, 0])
    world_max_x = np.max(points[:, 0])

    grid_x_size = int((world_max_x - world_min_x) / resolution) + 1
    grid_y_size = int((world_max_y - world_min_y) / resolution) + 1
    
    projected = np.full((grid_x_size, grid_y_size), fill_value=np.nan, dtype=np.float32)
    projected_x_indices = ((points[:, 0] - world_min_x) / resolution).astype(np.int32)
    projected_y_indices = ((points[:, 1] - world_min_y) / resolution).astype(np.int32)
    sums = np.zeros_like(projected)
    counts = np.zeros_like(projected, dtype=int)
    np.add.at(sums, (projected_x_indices, projected_y_indices), points[:, 2])
    np.add.at(counts, (projected_x_indices, projected_y_indices), 1)

    average_heightmap = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts!=0)
    projected = average_heightmap

    origin = (world_min_x, world_min_y)
    return projected, origin

def isolate_brightest_clusters(image, min_size=0.00):
    blobs = blob_log(image, min_sigma=.6, max_sigma=1.4, num_sigma=10, threshold=0.03)

    output = np.zeros_like(image)

    for blob in blobs:
        y, x, r = blob
        y = int(y)
        x = int(x)
        r = int(r)
        if r >= min_size:
            output[y, x] = image[y, x]

    # output = gaussian_laplace(image, sigma=1)
    return output

def generate_costmap(points, resolution=0.01, unknown_cost=255, lethal_magnitude=None, lethal_cost=254):
    projected, origin = project_point_cloud_optimized(points, resolution=resolution)

    gradient = np.gradient(projected, resolution, edge_order=1)
    magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)

    is_nan_mask = np.isnan(magnitude)
    mean_height = np.nanmean(magnitude)
    magnitude = np.nan_to_num(magnitude, nan=mean_height)

    if lethal_magnitude is not None:
        magnitude[magnitude >= lethal_magnitude] = lethal_cost
        magnitude[magnitude < lethal_cost] = 0
    # magnitude[is_nan_mask] = unknown_cost
    # lpg = gaussian_laplace(magnitude, sigma=2)
    return magnitude

def nvblox_costmap(points, resolution=0.01, unknown_cost=255, lethal_height=0.1, lethal_cost=254):
    projected, origin = project_point_cloud_optimized(points, resolution=resolution)

    med_height = np.nanmedian(projected)
    height_diff = projected - med_height
    height_diff[np.isnan(height_diff)] = 0.0
    height_diff[height_diff < 0] = 0.0

    costmap = np.zeros_like(height_diff)
    costmap[height_diff >= lethal_height] = lethal_cost

    return costmap

def plot_costmap(points, costmap, resolution=0.5):
    point_cloud = pv.PolyData(points)
    # point_cloud = point_cloud.reconstruct_surface()
    world_min_y = np.min(points[:, 1])
    world_max_y = np.max(points[:, 1])
    world_min_x = np.min(points[:, 0])
    world_max_x = np.max(points[:, 0])

    plotter = pv.Plotter()
    scalars = costmap[((points[:, 0] - world_min_x) // resolution).astype(int), ((points[:, 1] - world_min_y) // resolution).astype(int)]
    plotter.add_mesh(point_cloud, scalars=scalars, point_size=15, render_points_as_spheres=True, cmap='coolwarm')
    # plotter.add_mesh(point_cloud, point_size=15, render_points_as_spheres=True)
    plotter.set_background('white')
    plotter.show()


if __name__ == "__main__":
    resolution = 0.1

    start = time.time()
    data = np.load('point_cloud_archive/point_cloud_data.npz', allow_pickle=True)

    points = np.array(data['points'])

    projected, origin = project_point_cloud_optimized(points, resolution=resolution)
    plt.imshow(projected)
    plt.show()

    # projected = project_point_cloud_optimized(points, resolution=resolution)
    costmap = generate_costmap(points, resolution=resolution, lethal_magnitude=0.4)

    end = time.time()
    print(f"Time taken: {end - start} seconds")
    

    # costmap = np.zeros_like(costmap)
    plot_costmap(points, np.zeros_like(costmap), resolution=resolution)

    plt.imshow(costmap)
    plt.show()

    plot_costmap(points, costmap, resolution=resolution)


    