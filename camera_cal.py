import cv2
import numpy as np
import glob
import pickle

# The chessboard size is 9 by 6 corners
chess_cols = 9
chess_rows = 6

# Arrays to store corners' world coordinates and corresponding image coordinates
objpoints = []
imgpoints = []

# Generate world coordinates with Z = 0
objp = np.mgrid[0:chess_cols,0:chess_rows,0:1].T.reshape(-1,3).astype(np.float32)

# Create a list of calibration images filenames
images = glob.glob('./camera_cal/calibration*.jpg')

# Go through all images and find chessboard corners
for i, fname in enumerate(images):
    print("Processing image: ", fname)
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_cols, chess_rows), None)

    # Make sure that corners are added only when all of them are found
    if ret == True:
        print("Corners found!")
        objpoints.append(objp)
        imgpoints.append(corners)
        # Save images with chessboard corners drawn on
        img = cv2.drawChessboardCorners(img, (chess_cols, chess_rows), corners, ret)
        write_fname = './camera_cal/corners_found' + str(i+1) + '.jpg'
        cv2.imwrite(write_fname, img)

# Calibrate camera with collected corners
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

camera_cal = {}
camera_cal["mtx"] = mtx
camera_cal["dist"] = dist
with open('./camera_cal.p', 'wb') as pfile:
    pickle.dump(camera_cal, pfile)

