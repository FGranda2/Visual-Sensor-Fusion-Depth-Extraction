#!/usr/env/bin python3

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os

from utils import *


'''
Starter code for loading files, calibration data, and transformations
'''

# File paths
calib_dir = os.path.abspath('./data/calib')
image_dir = os.path.abspath('./data/image')
lidar_dir = os.path.abspath('./data/velodyne')
sample = '000000'

# Load the image
image_path = os.path.join(image_dir, sample + '.png')
image = img.imread(image_path)

# Load the LiDAR points
lidar_path = os.path.join(lidar_dir, sample + '.bin')
lidar_points = load_velo_points(lidar_path)

# Load the body to camera and body to LiDAR transforms
body_to_lidar_calib_path = os.path.join(calib_dir, 'calib_imu_to_velo.txt')
T_lidar_body = load_calib_rigid(body_to_lidar_calib_path)

# Load the camera calibration data
# Remember that when using the calibration data, there are 4 cameras with IDs
# 0 to 3. We will only consider images from camera 2.
lidar_to_cam_calib_path = os.path.join(calib_dir, 'calib_velo_to_cam.txt')
cam_to_cam_calib_path = os.path.join(calib_dir, 'calib_cam_to_cam.txt')
cam_calib = load_calib_cam_to_cam(lidar_to_cam_calib_path, cam_to_cam_calib_path)
intrinsics = cam_calib['K_cam2']
T_cam2_lidar = cam_calib['T_cam2_velo']

'''
For you to complete:
'''
# Part 1: Convert LiDAR points from LiDAR to body frame (for depths)
# Note that the LiDAR data is in the format (x, y, z, r) where x, y, and z are
# distances in metres and r is a reflectance value for the point which can be
# ignored. x is forward, y is left, and z is up. Depth can be calculated using
# d^2 = x^2 + y^2 + z^2

# First, we have T_lidar_body. We need T_body_lidar:
T_body_lidar = np.linalg.inv(T_lidar_body)
T_cam2_lidar[1, 0] = -T_cam2_lidar[1, 0]
T_cam2_lidar[1, 1] = -T_cam2_lidar[1, 1]
T_cam2_lidar[1, 2] = -T_cam2_lidar[1, 2]
T_cam2_lidar[1, 3] = T_cam2_lidar[1, 3]+0.7

# Then begin transformation of lidar points to body frame, and get depth values:
shape = lidar_points.shape
print(T_cam2_lidar)
Body_lidar_points = np.empty([shape[0], 4])
depths = np.empty([shape[0], 1])

for i in range(0, shape[0]):

    lidar_points[i, 3] = 1
    Body_lidar_points[i] = np.dot(T_body_lidar, lidar_points[i, 0:4].T).T
    depths[i] = np.sqrt(np.square(Body_lidar_points[i, 0]) + np.square(Body_lidar_points[i, 1]) +
                        np.square(Body_lidar_points[i, 2]))

# Part 2: Convert LiDAR points from LiDAR to camera 2 frame
# We already have the matrix from LiDAR to camera 2 frame, so compute transformations:
Cam2_lidar_points = np.empty([shape[0], 4])
for i in range(0, shape[0]):

    Cam2_lidar_points[i] = np.matmul(T_cam2_lidar, lidar_points[i, 0:4].T).T

# Part 3: Project the points from the camera 2 frame to the image plane. You
# may assume no lens distortion in the image. Remember to filter out points
# where the projection does not lie within the image field, which is 1242x375.
# First, make the projection to the image plane (With no lens distortion):

# Normalized image plane projection
Cam2_points = Cam2_lidar_points[0:shape[0], 0:3]
Norm_Cam2 = np.empty([shape[0], 3])

for i in range(0, shape[0]):

    Norm_Cam2[i] = np.divide(Cam2_points[i], Cam2_points[i, 2])

# Next, the pixel coordinates transform:
# Get the projection values in the image frame:
Im_points = np.empty([shape[0], 3])
for i in range(0, shape[0]):

    Im_points[i] = np.matmul(intrinsics, Norm_Cam2[i, 0:3].T).T

# Filter of points out of field of view:
X_d = Im_points[0:shape[0], 0]
Y_d = Im_points[0:shape[0], 1]

# First remove negative values
result = np.where(X_d < 0)
X_d = np.delete(X_d, result[0].T)
Y_d = np.delete(Y_d, result[0].T)
depths = np.delete(depths, result[0].T)

result = np.where(Y_d < 0)
X_d = np.delete(X_d, result[0].T)
Y_d = np.delete(Y_d, result[0].T)
depths = np.delete(depths, result[0].T)

# Next, remove values for X_d > 1242
result = np.where(X_d > 1242)
X_d = np.delete(X_d, result[0].T)
Y_d = np.delete(Y_d, result[0].T)
depths = np.delete(depths, result[0].T)

# Finally, remove values for Y_d > 375
result = np.where(Y_d > 375)
X_d = np.delete(X_d, result[0].T)
Y_d = np.delete(Y_d, result[0].T)
depths = np.delete(depths, result[0].T)

# Part 4: Overlay the points on the image with the appropriate depth values.
# Use a colormap to show the difference between points' depths and remember to
# include a colorbar.
extent = 0, 1242, 0, 375
print(np.amin(Y_d))
plt.figure()
plt.imshow(image, extent=extent)

#plt.scatter(Im_points[0:shape[0], 0], -(Im_points[0:shape[0], 1] - 3*np.amin(Y_d)), c=depths, alpha=0.1, marker=",")
plt.scatter(X_d, Y_d, c=depths, alpha=0.7, marker=",")
cb = plt.colorbar()
cb.set_label('Depth [meters]')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
plt.xlim(-1, 1)
plt.ylim(-1, 1)