from point_cloud_costmap import generate_costmap, plot_costmap, nvblox_costmap
from synthetic_data_generator import generate_terrain, generate_synthetic_point_cloud, visualize_point_cloud
import matplotlib.pyplot as plt
import numpy as np
import cv2

def calculate_f1_score(predicted_costmap, ground_truth_mask, lethal_threshold=254):
    predicted_obstacles = (predicted_costmap >= lethal_threshold).astype(int)
    tp = np.sum((predicted_obstacles == 1) & (ground_truth_mask == 1))
    fp = np.sum((predicted_obstacles == 1) & (ground_truth_mask == 0))
    fn = np.sum((predicted_obstacles == 0) & (ground_truth_mask == 1))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1_score

def compare_algorithms(terrain_kwargs, n=50):
    f1_grad_flow = 0
    f1_nvblox = 0
    for i in range(n):
        terrain, obstacle_mask = generate_terrain(**terrain_kwargs)
        synthetic_points = generate_synthetic_point_cloud(np.array(terrain), num_points=100000)

        costmap_grad_flow = generate_costmap(synthetic_points, resolution=0.1, lethal_magnitude=0.2)
        costmap_nvblox = nvblox_costmap(synthetic_points, resolution=0.1, lethal_height=0.05)

        obstacle_mask = cv2.resize(obstacle_mask.astype(np.uint8), costmap_nvblox.shape[::-1], interpolation=cv2.INTER_NEAREST)

        # Evaluate F1 score for gradient flow costmap
        f1_grad_flow += calculate_f1_score(costmap_grad_flow, obstacle_mask)
        f1_nvblox += calculate_f1_score(costmap_nvblox, obstacle_mask)

    return f1_grad_flow / n, f1_nvblox / n

if __name__ == "__main__":
    results = {}
    terrain_kwargs = {
        'xpix': 256,
        'ypix': 256,
        'scale': 5,
        'multi_level': False,
        'flat_ground': True,
        'above_ground_obstacles': True,
        'below_ground_obstacles': False
    }
    ## FLAT TERRAIN, ABOVE-GROUND OBSTACLES
    f1_grad_flow, f1_nvblox = compare_algorithms(terrain_kwargs, n=10)
    results['flat_terrain_above_ground'] = {
        'f1_grad_flow': f1_grad_flow,
        'f1_nvblox': f1_nvblox
    }

    ## Flat terrain, above and below ground obstacles
    terrain_kwargs['below_ground_obstacles'] = True
    f1_grad_flow, f1_nvblox = compare_algorithms(terrain_kwargs, n=10)
    results['flat_terrain_above_and_below'] = {
        'f1_grad_flow': f1_grad_flow,
        'f1_nvblox': f1_nvblox
    }

    ## Not flat terrain, obstacles
    terrain_kwargs['flat_ground'] = False
    f1_grad_flow, f1_nvblox = compare_algorithms(terrain_kwargs, n=10)
    results['non_flat_terrain'] = {
        'f1_grad_flow': f1_grad_flow,
        'f1_nvblox': f1_nvblox
    }

    ## Multi-level terrain, obstacles
    terrain_kwargs['multi_level'] = True
    terrain_kwargs['flat_ground'] = True
    f1_grad_flow, f1_nvblox = compare_algorithms(terrain_kwargs, n=10)
    results['multi_level_terrain'] = {
        'f1_grad_flow': f1_grad_flow,
        'f1_nvblox': f1_nvblox
    }

    print("Comparison Results:")
    for key, value in results.items():
        print(f"{key}: F1 Gradient Flow = {value['f1_grad_flow']}, F1 NVBlox = {value['f1_nvblox']}")

    
    # grouped bar chart
    labels = list(results.keys())
    grad_flow_scores = [results[key]['f1_grad_flow'] for key in labels]
    nvblox_scores = [results[key]['f1_nvblox'] for key in labels]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, grad_flow_scores, width, label='Gradient Flow')
    bars2 = ax.bar(x + width/2, nvblox_scores, width, label='NVBlox')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Costmap Generation Algorithm and Terrain Type')
    ax.legend()
    plt.xticks(x, labels)
    plt.show()
