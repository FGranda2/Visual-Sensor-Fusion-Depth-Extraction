#!/usr/env/bin python3

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os

from utils1 import *

# PART 1
# Question 1
# First, determine angles:
z_angle = np.radians(-90)
y_angle = np.radians(-23)
x_angle = np.radians(-10)

# Now, determine elementary rotation matrices:
C_x = np.array([[1, 0, 0], [0, np.cos(x_angle), -np.sin(x_angle)], [0, np.sin(x_angle), np.cos(x_angle)]])
C_x = np.around(C_x,3)
print('Matrix Cx is:\n', C_x)
C_y = np.array([[np.cos(y_angle), 0, np.sin(y_angle)], [0, 1, 0], [-np.sin(y_angle), 0, np.cos(y_angle)]])
C_y = np.around(C_y,3)
print('Matrix Cy is:\n', C_y)
C_z = np.array([[np.cos(z_angle), -np.sin(z_angle), 0], [np.sin(z_angle), np.cos(z_angle), 0], [0, 0, 1]])
# C_z = np.array([[0, 1, 0], [-1, 0, 0],[0, 0, 1]])
C_z = np.around(C_z,3)
print('Matrix Cz is:\n', C_z)

# Next, get the C_BL matrix
C_BL = np.dot(np.dot(C_z, C_y), C_x)
print('Matrix C_BL is:\n', np.around(C_BL, 3))

# Next, need the vector r_B, from the figure we get:
r_B = np.array([[2.57, -0.52, 1.32]])
print('Vector r_B is:\n', r_B)

# Finally form the transformation matrix:
T_BL = np.ones((4, 4))
T_BL[0:3, 0:3] = C_BL
T_BL[0:3, 3] = r_B
T_BL[3, 0:3] = 0
T_BL = np.around(T_BL, 3)
print('Matrix T_BL is:\n', T_BL)
print('The determinant is: \n', np.linalg.det(T_BL))

# Question 2
r_PL = np.array([[3.64, 8.30, 2.45, 1]]).T
print('The point in LIDAR frame is:\n', r_PL)
r_PB = np.dot(T_BL, r_PL)
r_PB = np.around(r_PB, 3)
print('The point in the body frame is:\n', r_PB[0:3, 0])

# Question 3
T_LB = np.linalg.inv(T_BL)
T_LB = np.around(T_LB, 3)
print('The Matrix T_LB is:\n', T_LB)

# PART 2
# Question 1
# State the angle of rotation
pitch_angle = np.radians(-90)

# Use the rotation matrix about pitch:
C_BC = np.array([[np.cos(pitch_angle), 0, -np.sin(pitch_angle)], [0, 1, 0], [np.sin(pitch_angle), 0, np.cos(pitch_angle)]])
C_BC = np.around(C_BC, 3)
print('Matrix C_BC is:\n', C_BC)

# Next, need the vector r_C, from the figure we get:
r_C = np.array([[2.82, 0.11, 1.06]])
print('Vector r_C is:\n', r_C)

# Finally form the transformation matrix:
T_BC = np.ones((4, 4))
T_BC[0:3, 0:3] = C_BC
T_BC[0:3, 3] = r_C
T_BC[3, 0:3] = 0
print('Matrix T_BC is:\n', T_BC)
print('The determinant is: \n', np.linalg.det(T_BC))

# Question 2
# The feature in body frame is:
P_B = np.array([[4.47, -0.206, 0.731, 1]]).T
# Now, use the inverse of T_BC to transform the point to the camera frame:
T_CB = np.linalg.inv(T_BC)
print('The matrix T_CB is:\n', np.around(T_CB, 3))
P_C = np.dot(T_CB, P_B)
print('The feature in the Camera frame is:\n', P_C[0:3, 0])

# Normalized image plane projection
P_C = P_C[0:3, 0]
Norm_P = np.divide(P_C,P_C[2])
print('Norm_P is:\n', np.around(Norm_P, 3))
# Plumb Bob Model distortion
# First, values of radial distortion:
K1 = -0.369
K2 = 0.197
K3 = 0.00135
X_n = Norm_P[0]
Y_n = Norm_P[1]
r = np.sqrt(np.square(X_n)+np.square(Y_n))
print('r is:\n', np.around(r, 3))
# Calculate radial distortion term:
Rad_dist = 1 + (K1 * np.power(r, 2)) + (K2 * np.power(r, 4)) + (K3 * np.power(r, 6))


# Values for tangential distortion:
T1 = 0.000568
T2 = -0.068

# Now calculate tangential distortion term:
Tang_dist = np.array([[2*T1*X_n*Y_n + (T2*(np.power(r, 2) + 2*np.power(X_n, 2))), 2*T2*X_n*Y_n + (T1*(np.power(r, 2) + 2*np.power(Y_n, 2)))]]).T

# Introduce lens distortion transform:
X_d = Rad_dist*X_n + Tang_dist[0]
Y_d = Rad_dist*Y_n + Tang_dist[1]
print('P_d:\n', np.around(X_d, 3), np.around(Y_d, 3))
# Next, the pixel coordinates transform values:
f_x = 959.79
f_y = 956.93
c_x = 696.02
c_y = 224.18

# Form matrix K:
K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
V_d = np.array([[X_d[0], Y_d[0], 1]]).T
V_s = np.dot(K,V_d)
print(np.around(V_s,0), np.around(V_s,3))

#Question 3
# We can see the feature is pointing to a STOP sign.