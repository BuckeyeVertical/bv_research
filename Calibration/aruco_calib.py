import numpy
import math
import cv2
from cv2 import aruco
import pickle
import glob
import os

# --- Board Definition ---
CHARUCOBOARD_ROWCOUNT = 3
CHARUCOBOARD_COLCOUNT = 4
# This dictionary seems to be working, so we'll stick with it.
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)

# --- VERY IMPORTANT ---
# Please double-check these values on the pattern generator website.
# The ratio between them is critical for corner interpolation.
SQUARE_LENGTH_METERS = 0.13329
MARKER_LENGTH_METERS = 0.09914

CHARUCO_BOARD = aruco.CharucoBoard(
        size=(CHARUCOBOARD_ROWCOUNT, CHARUCOBOARD_COLCOUNT),
        squareLength=SQUARE_LENGTH_METERS,
        markerLength=MARKER_LENGTH_METERS,
        dictionary=ARUCO_DICT)

# --- Image Collection ---
corners_all, ids_all = [], []
image_size = None
image_path_pattern = 'Calibration/images/**/*.jpg'
images = glob.glob(image_path_pattern, recursive=True)

print(f"Found {len(images)} images for calibration.")

# --- Main Loop with Visualization ---
for iname in images:
    print(f"--- Processing image: {iname}")
    img = cv2.imread(iname)
    if img is None:
        print("    -> Could not read image.")
        continue

    # Create a display copy to draw on
    display_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT)

    if ids is not None and len(ids) > 0:
        print(f"    -> Found {len(corners)} ArUco markers.")
        # Draw detected markers for visualization
        aruco.drawDetectedMarkers(display_img, corners, ids)
        
        # Try to interpolate corners
        resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=CHARUCO_BOARD)

        print(f"    -> interpolateCornersCharuco found {resp} corners.")

        # If any corners are found, draw them and save the results
        if resp > 0:
            aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids, (255, 0, 0))
            if resp > 10: # If it's a good detection, add it to our list
                corners_all.append(charuco_corners)
                ids_all.append(charuco_ids)
                if image_size is None: image_size = gray.shape[::-1]

        # --- VISUALIZATION ---
        # Scale image for display and show it
        scale_percent = 25 # Adjust scale if images are too large for your screen
        width = int(display_img.shape[1] * scale_percent / 100)
        height = int(display_img.shape[0] * scale_percent / 100)
        resized_img = cv2.resize(display_img, (width, height))
        
        cv2.imshow('ChArUco Detection Results', resized_img)
        # Press any key to continue to the next image. Press 'q' to quit.
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    else:
        print("    -> No ArUco markers detected.")

cv2.destroyAllWindows()

# --- Perform Final Calibration ---
if not corners_all:
    print("\nNo valid ChArUco boards were detected. Cannot perform calibration.")
else:
    print(f"\nStarting calibration with {len(corners_all)} valid images...")
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)
    
    if ret:
        print("\nCalibration successful!")
        print(f"Mean Reprojection Error: {ret:.4f} px")
        # ... (rest of the script is the same)