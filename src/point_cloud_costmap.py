import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import time

# this function projects a 3d pointcloud onto a 2d grid by averaging heights within a cell.
# optimized for speed using numpy's advanced indexing and accumulation functions.
# returns the projected 2d array and the origin (min x, min y) of the grid
def project_point_cloud_optimized(points, resolution=0.01):
    # Determine grid size and origin
    world_min_y = np.min(points[:, 1])
    world_max_y = np.max(points[:, 1])
    world_min_x = np.min(points[:, 0])
    world_max_x = np.max(points[:, 0])

    grid_x_size = int((world_max_x - world_min_x) / resolution) + 1
    grid_y_size = int((world_max_y - world_min_y) / resolution) + 1
    
    # Initialize projected grid
    projected = np.full((grid_x_size, grid_y_size), fill_value=np.nan, dtype=np.float32)
    sums = np.zeros_like(projected)
    counts = np.zeros_like(projected, dtype=int)

    # Calculate grid indices for each point
    projected_x_indices = ((points[:, 0] - world_min_x) / resolution).astype(np.int32)
    projected_y_indices = ((points[:, 1] - world_min_y) / resolution).astype(np.int32)

    # Accumulate sums and counts using advanced indexing
    np.add.at(sums, (projected_x_indices, projected_y_indices), points[:, 2])
    np.add.at(counts, (projected_x_indices, projected_y_indices), 1)

    # average the heights
    average_heightmap = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts!=0)
    projected = average_heightmap

    origin = (world_min_x, world_min_y)
    return projected, origin

# function to generate costmap from point cloud using gradient magnitude method
def generate_costmap(points, resolution=0.01, unknown_cost=255, lethal_magnitude=None, lethal_cost=254):
    # get ESDF projection
    projected, origin = project_point_cloud_optimized(points, resolution=resolution)

    # compute gradient
    gradient = np.gradient(projected, resolution, edge_order=1)
    magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)

    # handle NaN values by replacing them with the mean of the non-NaN magnitudes
    is_nan_mask = np.isnan(magnitude)
    mean_height = np.nanmean(magnitude)
    magnitude = np.nan_to_num(magnitude, nan=mean_height)

    # apply lethal threshold if provided
    if lethal_magnitude is not None:
        magnitude[magnitude >= lethal_magnitude] = lethal_cost
        magnitude[magnitude < lethal_cost] = 0
    return magnitude


# This function is used for comparison with nvblox costmap generation.
# while it skips over nvblox's ray traced TSDF generation, it mimics their costmap generation from a heightmap perspective.
# we were not comparing computational efficiency here, just costmap quality, which actually benefits nvblox, as the ray traced step sacrifices some accuracy for speed.
def nvblox_costmap(points, resolution=0.01, unknown_cost=255, lethal_height=0.1, lethal_cost=254):
    # get ESDF projection
    projected, origin = project_point_cloud_optimized(points, resolution=resolution)

    # compute median height and height differences
    med_height = np.nanmedian(projected)
    height_diff = projected - med_height
    height_diff[np.isnan(height_diff)] = 0.0

    # set negative height differences to zero
    height_diff[height_diff < 0] = 0.0
    costmap = np.zeros_like(height_diff)

    # use a uniform cost for heights above a vertical threshold.
    costmap[height_diff >= lethal_height] = lethal_cost

    return costmap

# function to visualize costmap on point cloud using pyvista
# uses pyvista mesh with each point in the pointcloud as a point in 3d space.
# colors the points based on the costmap values at their projected locations
def plot_costmap(points, costmap, resolution=0.5):
    point_cloud = pv.PolyData(points)
    # point_cloud = point_cloud.reconstruct_surface()
    world_min_y = np.min(points[:, 1])
    world_max_y = np.max(points[:, 1])
    world_min_x = np.min(points[:, 0])
    world_max_x = np.max(points[:, 0])

    plotter = pv.Plotter()
    # costmap color lookup based on point locations
    scalars = costmap[((points[:, 0] - world_min_x) // resolution).astype(int), ((points[:, 1] - world_min_y) // resolution).astype(int)]

    # add the mesh to the pyvista plotter, using the costmap values as scalars for coloring
    plotter.add_mesh(point_cloud, scalars=scalars, point_size=15, render_points_as_spheres=True, cmap='coolwarm')
    # plotter.add_mesh(point_cloud, point_size=15, render_points_as_spheres=True)
    plotter.set_background('white')
    plotter.show()


if __name__ == "__main__":
    resolution = 0.1

    start = time.time()
    # load recorded point cloud data
    data = np.load('point_cloud_archive/point_cloud_data.npz', allow_pickle=True)
    points = np.array(data['points'])

    # get and visualize esdf projection
    projected, origin = project_point_cloud_optimized(points, resolution=resolution)
    plt.imshow(projected)
    plt.show()

    # build a costmap using gradient magnitude method
    costmap = generate_costmap(points, resolution=resolution, lethal_magnitude=0.4)

    # build a costmap using nvblox method
    # costmap = nvblox_costmap(points, resolution=resolution, lethal_height=0.1)

    end = time.time()
    print(f"Time taken: {end - start} seconds")

    # visualize raw pointcloud
    plot_costmap(points, np.zeros_like(costmap), resolution=resolution)

    # visualize costmap
    plt.imshow(costmap)
    plt.show()

    # visualize costmap on point cloud
    plot_costmap(points, costmap, resolution=resolution)


    