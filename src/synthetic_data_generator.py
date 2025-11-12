from perlin_noise import PerlinNoise
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


# BUILD 2D IMAGE OF TERRAIN (PERLIN NOISE OR SOMETHING ELSE)
def sigmoid(x, steepness):
    return 1 / (1 + np.exp(-steepness * x))

def generate_terrain(xpix=256, ypix=256, scale=5,
                     flat_ground=False, above_ground_obstacles=True, below_ground_obstacles=True,
                     multi_level=False):
    terrain = []
    obstacle_mask = np.zeros((xpix, ypix))

    noise = PerlinNoise(octaves=1, seed=np.random.randint(0, 100))

    if flat_ground:
        terrain = np.zeros((xpix, ypix))
    else:
        for i in range(xpix):
            row = []
            for j in range(ypix):
                n = noise([i / (xpix * scale), j / (ypix * scale)])
                row.append(n)
            terrain.append(row)
        terrain = np.array(terrain)

    if multi_level:
        split_noise = PerlinNoise(octaves=2, seed=np.random.randint(0, 100))
        plateau_low = np.random.uniform(-0.2, 0.2)
        plateau_high = plateau_low + np.random.uniform(0.3, 0.6)
        steepness = np.random.uniform(100, 200)
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
    points = []
    xpix, ypix = image.shape
    for _ in range(num_points):
        x = np.random.uniform(0, xpix)
        y = np.random.uniform(0, ypix)
        ix = int(x)
        iy = int(y)
        z = image[ix, iy] * height_scale + np.random.normal(0, 0.01)
        points.append([x * world_size / xpix, y * world_size / ypix, z])
    return np.array(points)


# VISUALIZE POINT CLOUD
def visualize_point_cloud(points):
    cloud = pv.PolyData(points)
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=10)
    plotter.show()


if __name__ == "__main__":
    terrain, obstacle_mask = generate_terrain(xpix=256, ypix=256, scale=5, multi_level=True)

    synthetic_points = generate_synthetic_point_cloud(np.array(terrain), num_points=100000)

    plt.imshow(terrain)
    plt.show()

    visualize_point_cloud(synthetic_points)

    from point_cloud_costmap import project_point_cloud_optimized, generate_costmap, plot_costmap

    costmap = generate_costmap(synthetic_points, resolution=0.1, lethal_magnitude=0.3)

    plot_costmap(synthetic_points, costmap, resolution=0.1)