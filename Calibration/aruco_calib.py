# System information:
# - Linux Mint 18.1 Cinnamon 64-bit
# - Python 2.7 with OpenCV 3.2.0

import numpy
import math
import cv2
from cv2 import aruco
import pickle
import glob


# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5 
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=0.04,
        markerLength=0.02,
        dictionary=ARUCO_DICT)

# Collectors for corners and ids
corners_all = []   # 2D charuco corners per image
ids_all     = []   # corresponding corner IDs per image
image_size  = None # will be set from first valid image

# Gather all calibration images
images = glob.glob('data/calibration/*.jpg')

for iname in images:
    img = cv2.imread(iname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect aruco markers
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT)
    img     = aruco.drawDetectedMarkers(img, corners)

    # interpolate charuco corners
    resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=CHARUCO_BOARD)

    if resp > 20:
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)

        img = aruco.drawDetectedCornersCharuco(
            image=img,
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids)

        if image_size is None:
            image_size = gray.shape[::-1]

        # display for visual check
        prop = max(img.shape) / 1000.0
        disp = cv2.resize(img, (int(img.shape[1]/prop), int(img.shape[0]/prop)))
        cv2.imshow('Charuco board', disp)
        cv2.waitKey(0)
    else:
        print("Not able to detect a charuco board in image: {}".format(iname))

cv2.destroyAllWindows()

# sanity checks
if len(images) < 1:
    print("No images found for calibration.")
    exit()

if image_size is None:
    print("No valid charuco detections; adjust pattern size or image set.")
    exit()

# perform calibration
ret, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
    charucoCorners=corners_all,
    charucoIds=ids_all,
    board=CHARUCO_BOARD,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None)

print("Calibration RMS error reported by OpenCV: {:.4f}".format(ret))
print("Camera matrix:\n", cameraMatrix)
print("Distortion coeffs:\n", distCoeffs)

# ---- NEW: compute mean reprojection error manually ----
total_err   = 0.0
total_points = 0

for corners, ids, rvec, tvec in zip(corners_all, ids_all, rvecs, tvecs):
    # prepare 2D points
    img_pts = corners.reshape(-1, 2)

    # get corresponding 3D points from board
    idx = ids.flatten()  # shape (N,)
    obj_pts = CHARUCO_BOARD.chessboardCorners[idx]  # (N, 3)

    # project 3D points into image
    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec,
                                    cameraMatrix, distCoeffs)
    proj_pts = proj_pts.reshape(-1, 2)

    # sum of squared errors for this view
    err = cv2.norm(img_pts, proj_pts, cv2.NORM_L2)**2
    total_err   += err
    total_points += len(obj_pts)

mean_error = math.sqrt(total_err / total_points)
print("Mean reprojection error: {:.4f} pixels".format(mean_error))
# ---------------------------------------------------------

# save calibration
with open('calibration.pckl', 'wb') as f:
    pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)

print('Calibration successful. Saved to calibration.pckl')
