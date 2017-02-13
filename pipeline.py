import numpy as np
import cv2
import pickle
from image_processing import *

# Hardcode polygon coordinates in world space (src) and warped space (dst)
src = np.float32([[220,720],[1110,720],[735,480],[554,480]])
dst = np.float32([[320,720],[960,720],[960,0],[320,0]])
# Try to guestimate pixel space to world space dimensions
ym_per_pix = 25/720 # meters per pixel in y dimension (warped)
xm_per_pix = 3.7/640 # meters per pixel in x dimension (warped)
# Calculate perspective transform matrices
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

with open('./camera_cal.p', 'rb') as pfile:
    camera_cal = pickle.load(pfile)
    mtx = camera_cal['mtx']
    dist = camera_cal['dist']

def imread(fname):
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def warp(img):
    image_size = img.shape[0:2][::-1]
    warped = cv2.warpPerspective(img, M, image_size, flags=cv2.INTER_LINEAR)
    return warped

def unwarp(img):
    image_size = img.shape[0:2][::-1]
    unwarped = cv2.warpPerspective(img, Minv, image_size, flags=cv2.INTER_LINEAR)
    return unwarped

def draw_perspective(img, warped = False):
    imcopy = img.copy()
    pts = dst if warped else src
    pts = pts.astype(np.int32).reshape((-1,1,2))
    cv2.polylines(imcopy, [pts], True, (255,0,0), thickness=5)
    return imcopy

def edge_thresh(img):
    # 1) Preprocess image for edge detection (convert to grayscale and blur it)
    gray = preprocess_for_sobel(img)
    # 2) Perform sobel and directional thresholding
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=15, thresh=(10, 255))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=15, thresh=(20, 255))
    dir_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.35, 1.25))
    # 3) Close morphologically gradx and grady
    kernel = np.ones((5,5), np.uint8)
    gradx = cv2.morphologyEx(gradx, cv2.MORPH_CLOSE, kernel)
    grady = cv2.morphologyEx(grady, cv2.MORPH_CLOSE, kernel)
    # 4) Combine gradx and grady, where grady is filtered by direction
    combined = (dir_binary & grady) | gradx
    return combined

def color_thresh(img):
    # 1) Select yellow pixels
    hls_yellow = hls_select(img, low_thresh=(0, 100, 140), high_thresh=(35, 255, 255))
    # 2) Select white pixels
    hls_white = hls_select(img, low_thresh=(130, 100, 140), high_thresh=(180, 255, 255))
    # 3) Merge yellow and white pixels
    hls = hls_yellow | hls_white
    # 4) Close gaps by applying a morpohological operator
    kernel = np.ones((5,5), np.uint8)
    hls = cv2.morphologyEx(hls, cv2.MORPH_CLOSE, kernel)
    return hls

def curve_rad(leftx, lefty, rightx, righty, y_eval):
    fit_cr_l = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    fit_cr_r = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    fit_cr = (fit_cr_l + fit_cr_r) / 2
    A = fit_cr[0]
    B = fit_cr[1]
    radius = ((1 + (2*A*y_eval*ym_per_pix + B)**2)**1.5) / np.absolute(2*A)
    return int(radius)

def center_diff(left_fit, right_fit, bottom, midpoint):
    l = left_fit[0]*bottom*bottom + left_fit[1]*bottom + left_fit[2]
    r = right_fit[0]*bottom*bottom + right_fit[1]*bottom + right_fit[2]
    lane_center = (l + r) // 2
    return (midpoint - lane_center)*xm_per_pix

