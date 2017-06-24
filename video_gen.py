import cv2
import numpy as np
from image_processing import *
from pipeline import *
from lane import *
from moviepy.editor import VideoFileClip

last_left_fit = []
last_right_fit = []
debug = False
# Go through all images and generate a few other images 
def process_image(img):
    global last_left_fit, last_right_fit, debug
    # 1) Undistort the image
    undist = undistort(img)
    # 2) Edge threshold
    edges = edge_thresh(undist)
    # 3) Color threshold
    hls = color_thresh(undist)
    # 4) Apply a mask on edges and combine it with hls
    mask = np.zeros_like(undist)
    binary = hls
    if len(last_left_fit) > 0:
        draw_curve(mask, last_left_fit[-1], color=(255,255,255), thickness=40)
        draw_curve(mask, last_right_fit[-1], color=(255,255,255), thickness=40)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        binary = binary | (unwarp(mask) & edges)
    # 5) Warp the thresholded image
    bin_warped = warp(binary)
    # 6) Warp the image
    warped = warp(undist)
    # 7) Find centroids and pixels
    centroids = find_window_centroids(bin_warped,win_width=80,win_height=80,margin=100,minpix=500)
    leftx, lefty, rightx, righty = extract_lane_pixels(bin_warped,centroids,margin=100)
    full_scan = draw_window_centroids(bin_warped, centroids, leftx, lefty, rightx, righty, margin=100)
    # 8) Get the curve and smoothen it with historical data
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    last_left_fit = last_left_fit[-5:] + [left_fit]
    last_right_fit = last_right_fit[-5:] + [right_fit]
    left_fit = np.mean(np.array(last_left_fit), axis=0)
    right_fit = np.mean(np.array(last_right_fit), axis=0)
    last_left_fit[-1] = left_fit
    last_right_fit[-1] = right_fit
    # 9) Fast find pixels (unused)
    leftx, lefty, rightx, righty = fast_lane_extract(bin_warped, left_fit, right_fit, margin=100)
    fast_scan = draw_fitted_curve(bin_warped, leftx, lefty, rightx, righty)
    # 10) Draw lane and final image
    lane = draw_lane(undist, left_fit, right_fit)
    # 11) Find deviation from center
    diff = center_diff(left_fit, right_fit, lane.shape[0], lane.shape[1] // 2)
    from_center = 'left' if diff < 0 else 'right'
    # 12) Calculate curve radius from left_fit and right_fit
    mask = np.zeros_like(undist)
    draw_curve(mask, left_fit, color=(255,255,255), thickness=30)
    draw_curve(mask, right_fit, color=(255,255,255), thickness=30)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    leftx, lefty, rightx, righty = fast_lane_extract(mask, left_fit, right_fit, margin=100)
    radius = curve_rad(leftx, lefty, rightx, righty, lane.shape[0])
    # 13) Draw info on screen
    draw_header(lane, 130)
    draw_text(lane, "Radius of Curviture is {0}m".format(radius), pos = (50, 50))
    draw_text(lane, "Vehicle is {0:.2f}m {1} of center".format(abs(diff), from_center), pos = (50, 100))
    # 14) Prepare debug images
    if debug == False: return lane
    undist = cv2.resize(undist, (0,0), fx=0.33, fy=0.33)
    lane = cv2.resize(lane, (0,0), fx=0.33, fy=0.33)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges = cv2.resize(edges, (0,0), fx=0.33, fy=0.33)
    hls = cv2.cvtColor(hls, cv2.COLOR_GRAY2RGB)
    hls = cv2.resize(hls, (0,0), fx=0.33, fy=0.33)
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    binary = cv2.resize(binary, (0,0), fx=0.33, fy=0.33)
    warped = cv2.resize(warped, (0,0), fx=0.33, fy=0.33)
    bin_warped = cv2.cvtColor(bin_warped, cv2.COLOR_GRAY2RGB)
    bin_warped = cv2.resize(bin_warped, (0,0), fx=0.33, fy=0.33)
    full_scan = cv2.resize(full_scan, (0,0), fx=0.33, fy=0.33)
    fast_scan = cv2.resize(fast_scan, (0,0), fx=0.33, fy=0.33)
    top = np.concatenate((undist,lane,warped), axis=1)
    middle = np.concatenate((edges,hls,binary), axis=1)
    bottom = np.concatenate((bin_warped,full_scan,fast_scan), axis=1)
    result = np.concatenate((top,middle,bottom), axis=0)
    return result

input_video = 'project_video.mp4'
output_video = 'project_tracked.mp4'

clip1 = VideoFileClip(input_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)

