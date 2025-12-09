from perlin_noise import PerlinNoise
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

# sigmoid function.
def sigmoid(x, steepness):
    return 1 / (1 + np.exp(-steepness * x))

# full terrain generation function.
# handles flat vs noisy terrain, above and below ground obstacles, and multi-level terrain generation.
# this is done with layered perlin noise functions and a few other math tricks.
def generate_terrain(xpix=256, ypix=256, scale=5,
                     flat_ground=False, above_ground_obstacles=True, below_ground_obstacles=True,
                     multi_level=False):
    # Build a base terrain.
    terrain = []
    obstacle_mask = np.zeros((xpix, ypix)) # used for evaluation later
    noise = PerlinNoise(octaves=1, seed=np.random.randint(0, 100))
    if flat_ground:
        # flat terrain
        terrain = np.zeros((xpix, ypix))
    else:
        for i in range(xpix):
            # if not flat, build perlin noise terrain- gradual slopes
            row = []
            for j in range(ypix):
                n = noise([i / (xpix * scale), j / (ypix * scale)])
                row.append(n)
            terrain.append(row)
        terrain = np.array(terrain)

    # if multi level, add a steep plateau using a sigmoid function
    if multi_level:
        # determien location, steepness, and height of plateau randomly
        split_noise = PerlinNoise(octaves=2, seed=np.random.randint(0, 100))
        plateau_low = np.random.uniform(-0.2, 0.2)
        plateau_high = plateau_low + np.random.uniform(0.3, 0.6)
        steepness = np.random.uniform(100, 200)
        # add plateau to terrain
        for i in range(xpix):
            for j in range(ypix):
                n = split_noise([i / (xpix * scale), j / (ypix * scale)])

                sig_val = sigmoid(n, steepness)
                terrain[i][j] += (plateau_high - plateau_low) * sig_val + plateau_low

                is_transition = 0.2 < sig_val < 0.8
                if is_transition:
                    obstacle_mask[i][j] = 1
                

    # generate obstacles
    OBSTACLE_SCALE = 0.1
    noise2 = PerlinNoise(octaves=8, seed=np.random.randint(0, 100))
    # uses thresholded perlin noise to add obstacles.
    for i in range(xpix):
        for j in range(ypix):
            n = noise2([i / xpix, j / ypix])
            if n > 0.3 and above_ground_obstacles:
                terrain[i][j] += n * OBSTACLE_SCALE
                obstacle_mask[i][j] = 1
            elif n < -0.3 and below_ground_obstacles:
                terrain[i][j] += n * OBSTACLE_SCALE
                obstacle_mask[i][j] = 1

    return terrain, obstacle_mask

# SAMPLE POINT CLOUD FROM IMAGE
def generate_synthetic_point_cloud(image, num_points=250, height_scale=5.0, world_size=10.0):
    # this function randomly samples points from the terrain image to create a synthetic point cloud.
    # x and y are set based on pixel location, z is based on image intensity at that pixel.
    points = []
    xpix, ypix = image.shape
    for _ in range(num_points):
        # for every point, randomly sample a pixel
        x = np.random.uniform(0, xpix)
        y = np.random.uniform(0, ypix)
        ix = int(x)
        iy = int(y)
        # calculate z based on pixel intensity with some noise to simulate sensor noise
        z = image[ix, iy] * height_scale + np.random.normal(0, 0.01)
        # add this point to the pointcloud
        points.append([x * world_size / xpix, y * world_size / ypix, z])
    return np.array(points)


# VISUALIZE POINT CLOUD
def visualize_point_cloud(points):
    # visualize the point cloud using pyvista
    cloud = pv.PolyData(points)
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=10)
    plotter.show()


if __name__ == "__main__":
    terrain, obstacle_mask = generate_terrain(xpix=256, ypix=256, scale=5, multi_level=False, flat_ground=False,
                                             above_ground_obstacles=True, below_ground_obstacles=True)

    synthetic_points = generate_synthetic_point_cloud(np.array(terrain), num_points=100000)

    plt.imshow(terrain)
    plt.show()

    visualize_point_cloud(synthetic_points)

    from point_cloud_costmap import generate_costmap, plot_costmap, nvblox_costmap, project_point_cloud_optimized

    esdf = project_point_cloud_optimized(synthetic_points, resolution=0.1)[0]
    plt.imshow(esdf)
    plt.show()

    costmap = generate_costmap(synthetic_points, resolution=0.1, lethal_magnitude=0.3)
    # costmap = nvblox_costmap(synthetic_points, resolution=0.1, lethal_height=0.05)

    print(costmap.shape)

    plt.imshow(costmap)
    plt.show()

    plot_costmap(synthetic_points, costmap, resolution=0.1)