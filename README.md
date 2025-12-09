# Costmap Generation for Uneven Terrain using Gradient Signed Distance Fields

This is our final project for CSCI 5561, Computer Vision, Fall 2025.

Built and designed by Alex Berg and Sanaz Hosseini.

This project seeks to improve upon the works of Nvidia's nvblox, (“Nvblox.” Nvblox - Isaac_ros_docs Documentation, nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/index.html), a tool that is built for determining a costmap from a 3D environment represented as a pointcloud.

In this repository, we establish code for generating synthetic environments from perlin noise, as well as reading in real world models, represented as pointclouds, from .npz files. 

Post data intake, we project the point cloud onto a plane, compute a gradient of it using sobel kernels, and then use this gradient to build a final costmap.

We also include the raw code for using these costmaps in ROS2/ nav2. HOWEVER, all of the prerequisite config files and other data for a ROS2 integration of this code are NOT included in this repo. Those files can be found here: [UMN Lunabotics repository](https://github.com/GOFIRST-Robotics/Lunabotics). Members of this project are among the highest contributors on that project, and are responsible for all of the code related to this project within that repo. The ROS2 integration was meant for demonstration purposes and a proof of concept. It has a much more extensive setup process, which is documented in the Lunabotics repo, but will not be covered here as it is a multi-hour process. Instead, we provide .npz files with the raw pointclouds from the ROS integration for visualization and testing in this repo. For more info on how to run the lunabotics code, please contact Alex Berg at ber00221@umn.edu. 


## repo overview

This project consists of 3 directories:
* point_cloud_archive: contains a handful of real-world pointclouds extracted from the ROS integration.

