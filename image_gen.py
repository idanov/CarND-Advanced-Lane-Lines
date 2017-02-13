import cv2
import os
import numpy as np
import glob
from image_processing import *
from pipeline import *
from lane import *

# Create a list of calibration images filenames
images = glob.glob('./test_images/*.jpg')

# Go through all images and generate a few other images 
for i, fname in enumerate(images):
    print("Processing image: ", fname)
    # 0) Read image
    img = imread(fname)
    # 1) Undistort the image
    undist = undistort(img)
    # 2) Edge threshold
    edges = edge_thresh(undist)
    # 3) Color threshold
    hls = color_thresh(undist)
    # 4) Warp the thresholded image
    bin_warped = warp(hls)
    # 5) Warp the image
    warped = warp(undist)
    # 6) Find centroids and pixels
    centroids = find_window_centroids(bin_warped,win_width=80,win_height=80,margin=100,minpix=500)
    leftx, lefty, rightx, righty = extract_lane_pixels(bin_warped,centroids,margin=100)
    full_scan = draw_window_centroids(bin_warped, centroids, leftx, lefty, rightx, righty, margin=100)
    # 7) Fast find pixels
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    leftx, lefty, rightx, righty = fast_lane_extract(bin_warped, left_fit, right_fit, margin=100)
    fast_scan = draw_fitted_curve(bin_warped, leftx, lefty, rightx, righty)
    # 8) Draw lane and final image
    lane = draw_lane(undist, left_fit, right_fit)
    diff = center_diff(left_fit, right_fit, lane.shape[0], lane.shape[1] // 2)
    from_center = 'left' if diff < 0 else 'right'
    radius = curve_rad(leftx, lefty, rightx, righty, lane.shape[0])
    draw_header(lane, 130)
    draw_text(lane, "Radius of Curviture is {0}m".format(radius), pos = (50, 50))
    draw_text(lane, "Vehicle is {0:.2f}m {1} of center".format(abs(diff), from_center), pos = (50, 100))

    fname = os.path.basename(fname)
    # Save images with chessboard corners drawn on
    write_fname = './output_images/undist_' + fname
    imwrite(write_fname, undist)
    write_fname = './output_images/edges_' + fname
    imwrite(write_fname, edges)
    write_fname = './output_images/hls_' + fname
    imwrite(write_fname, hls)
    write_fname = './output_images/bin_warped_' + fname
    imwrite(write_fname, bin_warped)
    write_fname = './output_images/warped_' + fname
    imwrite(write_fname, warped)
    write_fname = './output_images/full_scan_' + fname
    imwrite(write_fname, full_scan)
    write_fname = './output_images/fast_scan_' + fname
    imwrite(write_fname, fast_scan)
    write_fname = './output_images/final_' + fname
    imwrite(write_fname, lane)

