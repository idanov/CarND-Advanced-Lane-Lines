import cv2
import numpy as np
from pipeline import *

def draw_curve(img, fit, color=(255,255,0), thickness=2):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    line = np.vstack((fitx, ploty)).astype(np.int32).T.reshape((-1,1,2))
    cv2.polylines(img, [line], False, color, thickness=thickness)

def paint_lane(img, left_fit, right_fit, color=(0,255,0)):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    left_line = np.vstack((left_fitx, ploty)).astype(np.int32).T.reshape((-1,1,2))
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    right_line = np.vstack((right_fitx, ploty)).astype(np.int32).T.reshape((-1,1,2))
    right_line = np.flipud(right_line)
    pts = np.vstack((left_line, right_line))
    cv2.fillPoly(img, [pts], color)

def draw_text(img, text, pos=(50, 50), color=(255,255,255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def find_window_centroids(warped, win_width=80, win_height=80, margin=100, minpix=500):
    # Get endpoint and midpoint for convenience
    endpoint = warped.shape[1]
    midpoint = warped.shape[1] // 2
    # Define initial centroids
    l_center = midpoint // 2
    r_center = midpoint + midpoint // 2
    # Define a convolution filter
    window = np.ones(win_width)
    window_centroids = []
    # Step through the windows one by one
    for level in range(warped.shape[0] // win_height):
        # Identify window boundaries in y
        win_top = warped.shape[0] - (level + 1) * win_height
        win_bottom = warped.shape[0] - level * win_height
        # Get the histogram
        histogram = np.sum(warped[win_top:win_bottom,:], axis=0)
        # Apply convolution over the histogram
        conv_signal = np.convolve(window, histogram, mode="same")
        # Find the left centroid (fallback to a previous value on failure)
        l_peak = np.argmax(conv_signal[:midpoint])
        if conv_signal[l_peak] > minpix:
            l_center = l_peak - margin
            l_center = l_center + np.argmax(histogram[l_center:l_center + 2 * margin])
        # Find the right centroid (fallback to a previous value on failure)
        r_peak = np.argmax(conv_signal[midpoint:])
        if conv_signal[midpoint + r_peak] > minpix:
            r_center = r_peak + midpoint - margin
            r_center = r_center + np.argmax(histogram[r_center:r_center + 2 * margin])

        # Update the midpoint
        midpoint = (l_center + r_center) // 2
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
        
    return window_centroids

def extract_lane_pixels(warped, centroids, margin=100):
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nzy = nonzero[0]
    nzx = nonzero[1]
    # Lambda for index extraction
    good = lambda t,b,l,r: ((nzy >= t) & (nzy < b) & (nzx >= l) & (nzx < r))
    # Deduce win height
    win_height = warped.shape[0] // len(centroids)
    # Create empty lists to receive left and right lane pixel indices
    llane_inds = good(0,0,0,0)
    rlane_inds = good(0,0,0,0)
    # Go through all levels
    for level in range(len(centroids)):
        # Identify window boundaries in y
        win_top = warped.shape[0] - (level + 1) * win_height
        win_bottom = warped.shape[0] - level * win_height
        # Get centroids for the current level
        l_center, r_center = centroids[level]
        
        lx_low = l_center - margin
        lx_high = l_center + margin
        # Identify the nonzero pixels in x and y within the window
        llane_inds = llane_inds | good(win_top, win_bottom, lx_low, lx_high)
        
        rx_low = r_center - margin
        rx_high = r_center + margin
        # Identify the nonzero pixels in x and y within the window
        rlane_inds = rlane_inds | good(win_top, win_bottom, rx_low, rx_high)

    # Extract left and right line pixel positions
    leftx = nzx[llane_inds]
    lefty = nzy[llane_inds] 
    rightx = nzx[rlane_inds]
    righty = nzy[rlane_inds]
    # Return results
    return leftx, lefty, rightx, righty

def draw_window_centroids(warped, centroids, leftx, lefty, rightx, righty, margin=100):
    out_img = np.dstack((warped, warped, warped))*255
    # Fit a second order polynomial to each line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Deduce win height
    win_height = warped.shape[0] // len(centroids)
    for level in range(len(centroids)):
        # Identify window boundaries in y
        win_top = warped.shape[0] - (level + 1) * win_height
        win_bottom = warped.shape[0] - level * win_height
        # Get centroids for the current level
        l_center, r_center = centroids[level]
        # Draw the windows on the visualization image
        # - left
        rx_low = r_center - margin
        rx_high = r_center + margin
        cv2.rectangle(out_img, (rx_low,win_top), (rx_high,win_bottom), (0,255,0), 2)
        # - right
        lx_low = l_center - margin
        lx_high = l_center + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (lx_low,win_top), (lx_high,win_bottom), (0,255,0), 2)

    # Paint pixels red and blue (left and right)
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # Calculate fitted curve and prepare it for drawing
    draw_curve(out_img, left_fit)
    draw_curve(out_img, right_fit)
    return out_img

def fast_lane_extract(warped, left_fit, right_fit, margin=100):
    nonzero = warped.nonzero()
    nzy = nonzero[0]
    nzx = nonzero[1]
    l_A, l_B, l_C = left_fit
    r_A, r_B, r_C = right_fit
    llane_inds = ((nzx > (l_A*(nzy**2) + l_B*nzy + l_C - margin)) & (nzx < (l_A*(nzy**2) + l_B*nzy + l_C + margin))) 
    rlane_inds = ((nzx > (r_A*(nzy**2) + r_B*nzy + r_C - margin)) & (nzx < (r_A*(nzy**2) + r_B*nzy + r_C + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nzx[llane_inds]
    lefty = nzy[llane_inds] 
    rightx = nzx[rlane_inds]
    righty = nzy[rlane_inds]
    return leftx, lefty, rightx, righty

def draw_fitted_curve(warped, leftx, lefty, rightx, righty, margin=100):
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped, warped, warped))*255
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Prepare curves for drawing
    # Window image
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    translation = np.array([0,0,margin])
    paint_lane(window_img, left_fit - translation, left_fit + translation)
    paint_lane(window_img, right_fit - translation, right_fit + translation)
    
    # Draw the lane onto the warped blank image
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    draw_curve(result, left_fit)
    draw_curve(result, right_fit)
    return result

def draw_lane(undist, left_fit, right_fit, half_line = 20):
    lane_img = np.zeros_like(undist)
    base_img = np.zeros_like(undist)
    translation = np.array([0,0,half_line])
    paint_lane(base_img, left_fit, right_fit, color=(0,0x99,0xcc))
    paint_lane(lane_img, left_fit-translation, left_fit+translation, color=(0,0,0xFF))
    paint_lane(lane_img, right_fit-translation, right_fit+translation, color=(0,0,0xFF))
    draw_curve(lane_img, (left_fit + right_fit) / 2, color = (0,0,0xAA), thickness=5)
    blended = cv2.addWeighted(base_img, 1.0, lane_img, 1.0, 0.0)
    blended = unwarp(blended)
    out_img = cv2.addWeighted(undist, 1.0, blended, 0.7, 0.0)
    return out_img

