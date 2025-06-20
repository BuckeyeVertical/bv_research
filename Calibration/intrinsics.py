import os
import cv2
import numpy as np

# Define the chessboard size (inner corners)
CHESSBOARD_SIZE = (7, 6)

# Prepare 3D points for a single chessboard image
objp = []
for i in range(CHESSBOARD_SIZE[1]):
    for j in range(CHESSBOARD_SIZE[0]):
        objp.append([j, i, 0])
objp = np.array(objp, dtype=np.float32)

obj_points = []
img_points = []

# Specify your folder here
image_folder = "Triangulation/images"

# List all files in the folder and filter out only JPGs (or PNGs)
all_files = os.listdir(image_folder)
images = [f for f in all_files if f.lower().endswith(".png")]

print(f"Found {len(images)} images in '{image_folder}':")
for idx, fname in enumerate(images):
    print(f"{idx+1}. {fname}")

img_size = None

for fname in images:
    # Build the full path
    full_path = os.path.join(image_folder, fname)
    img = cv2.imread(full_path)
    if img is None:
        print(f"Could not read image '{full_path}'—skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = (gray.shape[1], gray.shape[0])

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if found:
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11,11),
            (-1,-1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        obj_points.append(objp)
        img_points.append(corners_refined)

        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners_refined, found)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(300)

cv2.destroyAllWindows()

if img_size is not None and len(obj_points) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        img_size,
        None,
        None
    )
    print("\nCamera Matrix:\n", camera_matrix)
    print("\nDistortion Coeffs:\n", dist_coeffs)
    print("\nReprojection Error:\n", ret)
else:
    print("\nNo valid images or no chessboard corners found. Calibration skipped.")
