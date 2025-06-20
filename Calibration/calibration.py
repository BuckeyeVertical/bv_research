import numpy as np
import cv2 as cv
import glob
import pickle


chessboardSize = (10,7)
frameSize = (2056,1538)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

objpoints = []
imgpoints = [] 
images = glob.glob('Triangulation/images/*.bmp')
print(f"Found {len(images)} images.")
for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    print("corners method was called")
    #if found add object points and image points 
    if ret == True:
        print("found corners")
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        #draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()
print("Calibrating Camera...")
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

print(cameraMatrix)

# [[9.31696166e+03 0.00000000e+00 9.42461549e+02]
#  [0.00000000e+00 1.02161568e+04 9.53746283e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]