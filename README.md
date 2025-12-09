# Costmap Generation for Uneven Terrain using Gradient Signed Distance Fields

This is our final project for CSCI 5561, Computer Vision, Fall 2025.

Built and designed by Alex Berg and Sanaz Hosseini.

This project seeks to improve upon the works of Nvidia's nvblox (“Nvblox.” Nvblox - Isaac_ros_docs Documentation, nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/index.html), a tool that is built for determining a costmap from a 3D environment represented as a pointcloud.

In this repository, we provide code for generating synthetic environments from Perlin noise, as well as reading in real-world models represented as point clouds from .npz files. 

Post data intake, we project the point cloud onto a plane, compute a gradient of it using sobel kernels, and then use this gradient to build a final costmap.

We also include the raw code for using these costmaps in ROS2/ nav2. HOWEVER, all the prerequisite config files and other data for a ROS2 integration of this code are NOT included in this repo. Those files can be found here: [UMN Lunabotics repository](https://github.com/GOFIRST-Robotics/Lunabotics). Members of this project are among the highest contributors on that project and are responsible for all of the code related to this project within that repo. The ROS2 integration was meant for demonstration purposes and a proof of concept. It has a much more extensive setup process, which is documented in the Lunabotics repo, but will not be covered here as it is a multi-hour process. Instead, we provide .npz files with the raw pointclouds from the ROS integration for visualization and testing in this repo. For more info on how to run the lunabotics code, please contact Alex Berg at ber00221@umn.edu. 


## repo overview
This project consists of 3 directories:
* point_cloud_archive: contains a handful of real-world point clouds extracted from the ROS integration.

* ros2_code: Includes demo code for a ROS2 integration. This is the same code present in the lunabotics repo. limited to our contributions rather than the entire repo. This code will not run outside of ros2.

* src: contains the code for visualization, testing, evaluation, etc.

In the src directory, you will find 3 important files: metrics.py, point_cloud_costmap.py, and synthetic_data_generator.py, which handle an automated comparison of our tool vs nvblox, costmap generation from a specified real-world recording, and costmap generation on synthetic data, respectively. 

## setup and prerequisites
This code requires little setup. It was run and tested on Python 3.12, on a low-end consumer desktop (R5 2600x, single core, 16GB DDR4 RAM)

Dependencies can be installed through the provided requirements.txt, or by manual installation of the following:
- opencv-python==4.12.0
- numpy==1.26.4
- pyvista==0.46.3
- matplotlib==3.10.3
- perlin-noise==1.13

## Running Code
As a result of the testing and development process, each of the mentioned scripts has a primary function that performs a distinct visualization task, making it easy to run.
Each of the files described above can be run independently and will open pyvista or matplotlib visualizations of their outputs automatically. 

To run and see the metrics that compare our method to the nvblox algorithm, run `python src/metrics.py`

To run and see a costmap made from a real-world environment, run `python src/point_cloud_costmap.py`

To run and see a costmap made from a virtual environment, run `python src/synthetic_data_generation.py`

All of these commands will run on their own, with no further required user input. If you'd like to tweak parameters or settings on these results, open the respective file, scroll to the main function at the bottom. All of the easily adjustable settings are adjustable via function parameters, and are labeled self-explantorily. For even more detail on how to run, consult Alex Berg at ber00221@umn.edu, or refer to the demo video as we discuss most of these settings.
