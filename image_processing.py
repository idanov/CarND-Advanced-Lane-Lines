import numpy as np
import cv2

def preprocess_for_sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    assert len(gray.shape) == 2, "Make sure that the image is grayscale"
    # Apply the following steps to img
    # 1) Take the derivative in x or y given orient = 'x' or 'y'
    dx = 1 if orient=='x' else 0
    dy = 1 if orient=='y' else 0
    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize = sobel_kernel)
    # 2) Take the absolute value of the derivative or gradient
    sobel = np.absolute(sobel)
    # 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel = np.uint8(255 * sobel / np.max(sobel))
    # 4) Create a mask of 1's where the scaled gradient magnitude 
    binary_output = np.zeros_like(sobel)
    binary_output[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1
            # is > thresh_min and < thresh_max
    # 5) Return this mask as your binary_output image
    return binary_output

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    assert len(gray.shape) == 2, "Make sure that the image is grayscale"
    # 1) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 2) Calculate the magnitude
    magn = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 3) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    magn_scaled = np.uint8(255 * magn/np.max(magn))
    # 4) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(magn_scaled)
    binary_output[(magn_scaled >= mag_thresh[0]) & (magn_scaled <= mag_thresh[1])] = 1
    # 5) Return this mask as your binary_output image
    return binary_output

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    assert len(gray.shape) == 2, "Make sure that the image is grayscale"
    # 1) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 2) Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    # 3) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    arctan = np.arctan2(abs_sobely, abs_sobelx)
    # 4) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(arctan).astype(np.uint8)
    binary_output[(arctan >= thresh[0]) & (arctan <= thresh[1])] = 1
    # 5) Return this mask as your binary_output image
    return binary_output

def hls_select(img, low_thresh=(0, 0, 0), high_thresh=(255,255,255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Split channels
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    # 3) Apply a threshold to each channel
    h_idx = (H > low_thresh[0]) & (H <= high_thresh[0])
    l_idx = (L > low_thresh[1]) & (L <= high_thresh[1])
    s_idx = (S > low_thresh[2]) & (S <= high_thresh[2])
    # 4) Return a binary mask of the threshold result
    binary_output = np.zeros_like(H)
    binary_output[h_idx & l_idx & s_idx] = 1
    return binary_output

