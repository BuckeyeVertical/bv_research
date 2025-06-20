import cv2
import numpy as np

#intrinsic matrix
K = np.array([[9.31696166e+03, 0.00000000e+00, 9.42461549e+02],
              [0.00000000e+00, 1.02161568e+04, 9.53746283e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

#extrinsic parameters (top-down orientation)
#this also might be the identity matrix not sure yet
R = np.array([[1, 0, 0],
              [0, -1, 0],
              [0, 0, -1]], dtype=np.float32)
#altitude meters
h = 30.0 
#displacement meters
baseline = 5.0

#projection matrices one per image
#K[R|t]
#t1 = [0, 0, altitude]
#t2 = [baseline, 0, altitude]
P1 = K @ np.hstack((R, [[0], [0], [h]]))
P2 = K @ np.hstack((R, [[baseline], [0], [h]]))

#load corresponding points example 4 points which are corresponding corners of object
points1 = np.array([[200, 300], [400, 500], [150, 600], [700, 200]], dtype=np.float32).T
points2 = np.array([[180, 310], [380, 510], [130, 610], [680, 210]], dtype=np.float32).T

#triangulate
points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
#points_4d = cv2.triangulatePoints(points_4d, P3, pts3)  # Add third view
points_3d = (points_4d[:3, :] / points_4d[3, :]).T

#Triangulate using all pairwise combinations
#points_3d_12 = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
#points_3d_13 = cv2.triangulatePoints(P1, P3, pts1.T, pts3.T)

#Average results for better accuracy
#inal_3d = (points_3d_12 + points_3d_13) / 2



#convert to ground coordinates (Z is depth below drone)
#not sure if we need this
ground_coords = np.column_stack((points_3d[:, 0], points_3d[:, 1], h - points_3d[:, 2]))

print("Reconstructed 3D Ground Coordinates:\n", ground_coords)