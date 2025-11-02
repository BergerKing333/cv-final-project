from perlin_noise import PerlinNoise
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

xpix, ypix = 256, 256
SCALE = 5
terrain = []

# BUILD 2D IMAGE OF TERRAIN (PERLIN NOISE OR SOMETHING ELSE)


# base terrain
noise = PerlinNoise(octaves=1, seed=np.random.randint(0, 100))

for i in range(xpix):
    row = []
    for j in range(ypix):
        n = noise([i / (xpix * SCALE), j / (ypix * SCALE)])
        row.append(n)
    terrain.append(row)

terrain = np.array(terrain)

# generate obstacles
OBSTACLE_SCALE = 0.2
noise2 = PerlinNoise(octaves=8, seed=np.random.randint(0, 100))
for i in range(xpix):
    for j in range(ypix):
        n = noise2([i / xpix, j / ypix])
        if n > 0.3:
            terrain[i][j] += n * OBSTACLE_SCALE
        elif n < -0.3:
            terrain[i][j] += n * OBSTACLE_SCALE


plt.imshow(terrain)
plt.show()

# SAMPLE POINT CLOUD FROM IMAGE
def generate_synthetic_point_cloud(image, num_points=250, height_scale=5.0, world_size=10.0):
    points = []
    xpix, ypix = image.shape
    for _ in range(num_points):
        x = np.random.uniform(0, xpix)
        y = np.random.uniform(0, ypix)
        ix = int(x)
        iy = int(y)
        z = image[ix, iy] * height_scale + np.random.normal(0, 0.1)
        points.append([x * world_size / xpix, y * world_size / ypix, z])
    return np.array(points)

synthetic_points = generate_synthetic_point_cloud(np.array(terrain), num_points=20000)

# VISUALIZE POINT CLOUD
def visualize_point_cloud(points):
    cloud = pv.PolyData(points)
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=10)
    plotter.show()

visualize_point_cloud(synthetic_points)