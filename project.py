#!/usr/bin/env python3

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os
import math
import pickle
import pdb
import sys

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

distortion_coeffs_pickle_file = "camera_cal/dist_pickle.p"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", '--verbose', action='store_true', help="be verbose")
parser.add_argument("-c", '--calibrate', action='store_true', help="determine camera calibration")
parser.add_argument("-n", '--num_images', type=int, help="number of images to process")
parser.add_argument("-f", '--image_files', type=str, help="file(s) to process, uses glob")
parser.add_argument("-m", '--video', action='store_true', help="process video instead of images")
args = parser.parse_args()

g_debug_internal = False
g_error_frames = 0


# The goals / steps of this project are the following:
# 
# X 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# X 2. Apply a distortion correction to raw images.
# X 3. Use color transforms, gradients, etc., to create a thresholded binary image.
# X 4. Apply a perspective transform to rectify binary image ("birds-eye view").
# X 5. Detect lane pixels and fit to find the lane boundary.
# X 6. Determine the curvature of the lane and vehicle position with respect to center.
# X 7. Warp the detected lane boundaries back onto the original image.
# X 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 

class Subplotter:
    # Handle subplots intelligently.
    def __init__(self):
        self.cols = 0
        self.rows = 0
        self.current = 0

    def setup(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.current = 1
        # Plot the result
        f, ax = plt.subplots(self.rows, self.cols, figsize=(14, 8))
        f.tight_layout()

    def next(self, image, title=None, just_plot=False):
        if self.current == 0:
            print("ERROR: subplot next called before setup")
            sys.exit(1)

        if self.current > (self.cols * self.rows):
            print("ERROR: too many subplots for rows, cols:", self.rows, self.cols)
            sys.exit(1)

        plt.subplot(self.rows, self.cols, self.current)

        if just_plot:
            plt.plot(image)
        else:
            plt.imshow(image.squeeze(), cmap='gray')

        if title:
            plt.title(title)
        self.current = self.current + 1

    def show(self):
        #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

g_subplotter = Subplotter()

def imshow_full_size(img, title=False, *args, **kwargs):
    dpi = 100
    margin = 0.05 # (5% of the width/height of the figure...)
    ypixels, xpixels = img.shape[0], img.shape[1]
    
    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * xpixels / dpi, (1 + margin) * ypixels / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    ax.imshow(img, interpolation='none', *args, **kwargs)
    if title:
        plt.title(title)
    plt.show()

class ImageUndistorter:
# 2. Apply a distortion correction to raw images.
    def __init__(self):
        self.init = False
        self.mtx = None
        self.dist = None

    def undistort(self, image):
        if self.init == False:
            # Read in the saved distortion coefficients only once
            dist_pickle = pickle.load( open( distortion_coeffs_pickle_file, "rb" ) )
            self.mtx = dist_pickle["mtx"]
            self.dist = dist_pickle["dist"]
            self.init = True

        undistorted_image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        return undistorted_image

def imshow_full_size(img, title=False, *args, **kwargs):
    dpi = 100
    margin = 0.05 # (5% of the width/height of the figure...)
    ypixels, xpixels = img.shape[0], img.shape[1]
    
    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * xpixels / dpi, (1 + margin) * ypixels / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    ax.imshow(img, interpolation='none', *args, **kwargs)
    if title:
        plt.title(title)
    plt.show()

# 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
def calibrate_camera():
    if args.verbose:
        print("Determining camera calibration and distortion coefficients")

    chessboard_num_rows = 6
    chessboard_num_cols = 9

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_num_rows*chessboard_num_cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_num_cols, 0:chessboard_num_rows].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    image_filenames = glob.glob('camera_cal/*.jpg')

    for image_filename in image_filenames:
        if args.verbose:
            print(image_filename)
        image = cv2.imread(image_filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #plt.imshow(gray, cmap='gray')
        #plt.show()

        ret, corners = cv2.findChessboardCorners(gray, (chessboard_num_cols,chessboard_num_rows), None)

        if ret == True:
            if args.verbose:
                print("Found chessboard corners")
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            #color = cv2.drawChessboardCorners(image, (chessboard_num_cols,chessboard_num_rows), corners, ret)
            #plt.imshow(color)
            #plt.show()
        else:
            print("Can't find chessboard corners in calibration image:", image_filename)

    # Test undistortion on an image
    test_image_filename = 'camera_cal/calibration2.jpg'
    img = cv2.imread(test_image_filename)
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)

    test_image_output_filename = 'camera_cal/calibration2-undist.jpg'
    cv2.imwrite(test_image_output_filename,dst)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( distortion_coeffs_pickle_file, "wb" ) )
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30) 
    plt.show()

def test_undistortion():
    #img = cv2.imread('test_images/test1.jpg')
    img = cv2.imread('camera_cal/calibration2.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_undistorter = ImageUndistorter()
    dst = image_undistorter.undistort(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30) 
    plt.savefig("calibration2-undistorted.png", bbox_inches='tight')
    plt.show()

def raw_threshold(image, s_min, s_max):
    thresholded = np.zeros_like(image)
    thresholded[(image >= s_min) & (image <= s_max)] = 1
    return thresholded

def plot_colors(rgb_image):
    r_channel = rgb_image[:,:,0]
    g_channel = rgb_image[:,:,1]
    b_channel = rgb_image[:,:,2]

    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
    hls_h_channel = hls[:,:,0]
    hls_l_channel = hls[:,:,1]
    hls_s_channel = hls[:,:,2]

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv_h_channel = hsv[:,:,0]
    hsv_s_channel = hsv[:,:,1]
    hsv_v_channel = hsv[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # test_images/straight_lines1.jpg
    #   R, HLS_L, HLS_S, HSV_S, HSV_V
    # test_images/straight_lines2.jpg
    #   R, G, HLS_L, HLS_S, HSV_V
    # test_images/test1.jpg
    #   R, B, HLS_S, HSV_S, HSV_V
    # test_images/test2.jpg
    #   R, G, HLS_L, HLS_S, HSV_S, HSV_V
    # test_images/test3.jpg
    #   R, G, HLS_L, HLS_S, HSV_S, HSV_V
    # test_images/test4.jpg
    #   R, HLS_S, HSV_S, HSV_V
    # test_images/test5.jpg
    #   R, HLS_S, HSV_V
    # test_images/test6.jpg
    #   R, G, HLS_L, HLS_S, HSV_V

    # Total counts: (out of 8 images)
    #     R: 8
    #     G: 4
    #     B: 1
    # HLS_H: 0
    # HLS_L: 5
    # HLS_S: 8
    # HSV_H: 0
    # HSV_S: 5
    # HSV_V: 8
    # So remove B, HLS_H, HSV_H
    # Then HSV_S is washed out in several, eliminate it
    # G isn't any better than R and sometimes worse, eliminate G

    if True:
        g_subplotter.setup(cols=3,rows=3)
        #g_subplotter.next(rgb_image, 'original')
        #g_subplotter.next(gray, 'gray')
        #g_subplotter.next(gray, 'gray2')
        g_subplotter.next(r_channel, 'R')
        g_subplotter.next(g_channel, 'G')
        g_subplotter.next(b_channel, 'B')
        #g_subplotter.next(hls_h_channel, 'HLS_H')
        g_subplotter.next(rgb_image, 'orig')
        g_subplotter.next(hls_l_channel, 'HLS_L')
        g_subplotter.next(hls_s_channel, 'HLS_S')
        g_subplotter.next(hsv_h_channel, 'HSV_H')
        g_subplotter.next(hsv_s_channel, 'HSV_S')
        g_subplotter.next(hsv_v_channel, 'HSV_V')
        g_subplotter.show()

    if False:
        # test_images/test1.jpg: light gray road, yellow + white lines
        #   HLS_S is the only one that works, but it misses some white lines
        #   R gives good white line visibility
        thresh_min=100
        thresh_max=255
        g_subplotter.setup(cols=5,rows=3)
        g_subplotter.next(gray, 'gray')
        g_subplotter.next(r_channel, 'R')
        g_subplotter.next(hls_l_channel, 'HLS_L')
        g_subplotter.next(hls_s_channel, 'HLS_S')
        g_subplotter.next(hsv_v_channel, 'HSV_V')

        g_subplotter.next(np.zeros_like(gray), '')
        g_subplotter.next(np.ones_like(gray), '')
        g_subplotter.next(rgb_image, 'original')
        g_subplotter.next(np.ones_like(gray), '')
        g_subplotter.next(np.ones_like(gray), '')

        g_subplotter.next(raw_threshold(gray,thresh_min,thresh_max), 'gray')
        g_subplotter.next(raw_threshold(r_channel,thresh_min,thresh_max), 'R')
        g_subplotter.next(raw_threshold(hls_l_channel,thresh_min,thresh_max), 'HLS_L')
        g_subplotter.next(raw_threshold(hls_s_channel,thresh_min,thresh_max), 'HLS_S')
        g_subplotter.next(raw_threshold(hsv_v_channel,thresh_min,thresh_max), 'HSV_V')

        g_subplotter.show()

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    dx = 1
    dy = 0
    if orient == 'y':
        dx = 0
        dy = 1
    sobel_deriv = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=sobel_kernel)
    abs_sobel_deriv = np.absolute(sobel_deriv)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel_deriv/np.max(abs_sobel_deriv))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=sobel_kernel)
    
    # 3) Calculate the magnitude 
    mag_sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=sobel_kernel)
    
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobel_x)
    abs_sobely = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir)
    binary_output[(dir >= thresh[0]) & (dir <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def color_threshold(image, thresh=(0,255)):
    s_binary = np.zeros_like(image)
    s_binary[(image >= thresh[0]) & (image <= thresh[1])] = 1
    return s_binary

# 4. Apply a perspective transform to rectify binary image ("birds-eye view").
def create_top_down_image(orig_image, binary_image):
    # Source images are 1280 wide by 720 high.
    # Grab the image shape
    img_height = binary_image.shape[0]
    img_width  = binary_image.shape[1]
    img_size = (img_width, img_height)

    top_of_mask = 450
    mask_lower_left  = ( 120, img_height-1)
    mask_upper_left  = ( 575, top_of_mask)
    mask_upper_right = ( 725, top_of_mask)
    mask_lower_right = (1200, img_height-1)

    # Keep only the part of the image likely to have lanes on it: right in front of the car.
    # (Keeps only the middle/bottom part of the image.)
    bounding_shape = np.array([[mask_lower_left, mask_upper_left, mask_upper_right, mask_lower_right]], dtype=np.int32)
    binary_image_copy = np.copy(binary_image)
    masked_binary_image = region_of_interest(binary_image, bounding_shape)



    src_top_right_x = 700 - 10
    src_top_y = 448
    src_bottom_right_x = 1130 + 50
    src_bottom_y = 719
    src_bottom_left_x = 180 - 50
    src_top_left_x = 580 + 10

    # For source points I'm choosing a trapezoid around straight lane lines
    # in the center of the camera.
    src = np.float32([ 
                      [src_top_right_x,    src_top_y],# upper right
                      [src_bottom_right_x, src_bottom_y],# lower right
                      [src_bottom_left_x,  src_bottom_y],# lower left
                      [src_top_left_x,     src_top_y],# upper left
                      ])
    if 0:
        lines_on_orig_image = np.copy(orig_image)
        #cv2.line(lines_on_orig_image, tuple(src[0]), tuple(src[1]), color=[255,0,0], thickness=1)
        #cv2.line(lines_on_orig_image, tuple(src[1]), tuple(src[2]), color=[255,0,0], thickness=1)
        #cv2.line(lines_on_orig_image, tuple(src[2]), tuple(src[3]), color=[255,0,0], thickness=1)
        #cv2.line(lines_on_orig_image, tuple(src[3]), tuple(src[0]), color=[255,0,0], thickness=1)
        cv2.line(lines_on_orig_image, mask_lower_left, mask_upper_left, color=[255,0,0], thickness=3)
        cv2.line(lines_on_orig_image, mask_upper_left, mask_upper_right, color=[255,0,0], thickness=3)
        cv2.line(lines_on_orig_image, mask_upper_right, mask_lower_right, color=[255,0,0], thickness=3)
        cv2.line(lines_on_orig_image, mask_lower_right, mask_lower_left, color=[255,0,0], thickness=3)
        g_subplotter.setup(cols=2,rows=2)
        g_subplotter.next(lines_on_orig_image, 'orig')
        g_subplotter.next(binary_image_copy, 'binary orig')
        g_subplotter.next(masked_binary_image, 'masked binary')
        g_subplotter.show()

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst_left_x = 300
    dst_right_x = 900
    dst_top_y = 0
    dst_bottom_y = 719
    dst = np.float32([
                      [dst_right_x, dst_top_y],# upper right
                      [dst_right_x, dst_bottom_y],# lower right
                      [dst_left_x,  dst_bottom_y],# lower left
                      [dst_left_x,  dst_top_y],# upper left
                      ])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    top_down = cv2.warpPerspective(masked_binary_image, M, img_size)

    top_down_color = cv2.warpPerspective(orig_image, M, img_size)

    if 0:
        lines_on_transformed_image = np.copy(top_down_color)
        cv2.line(lines_on_transformed_image, tuple(dst[0]), tuple(dst[1]), color=[255,0,0], thickness=1)
        cv2.line(lines_on_transformed_image, tuple(dst[1]), tuple(dst[2]), color=[255,0,0], thickness=1)
        cv2.line(lines_on_transformed_image, tuple(dst[2]), tuple(dst[3]), color=[255,0,0], thickness=1)
        cv2.line(lines_on_transformed_image, tuple(dst[3]), tuple(dst[0]), color=[255,0,0], thickness=1)

        # Line up the green line with the outer lane line edges to verify they're straight.
        delta_right_x = -40
        delta_left_x = 40
        cv2.line(lines_on_transformed_image,
                 (dst_right_x+delta_right_x, dst_top_y), (dst_right_x+delta_right_x, dst_bottom_y),
                 color=[0, 255,0], thickness=1)
        cv2.line(lines_on_transformed_image,
                 (dst_left_x+delta_left_x, dst_top_y), (dst_left_x+delta_left_x, dst_bottom_y),
                 color=[0, 255,0], thickness=1)
        g_subplotter.next(lines_on_transformed_image, 'trans')
        g_subplotter.next(binary_image, 'orig_binary')
        g_subplotter.next(top_down, 'top_down_binary')
        g_subplotter.show()

    return top_down, Minv

# 5. Detect lane pixels and fit to find the lane boundary.
def find_lane_lines_basic(orig_image, top_down_binary_image, Minv):
    global g_debug_internal
    global g_error_frames
    img_height = top_down_binary_image.shape[0]
    img_width = top_down_binary_image.shape[1]
    # Assuming you have created a warped binary image called "top_down_binary_image"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(top_down_binary_image[img_height//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((top_down_binary_image, top_down_binary_image, top_down_binary_image))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    if 0:
        g_subplotter.setup(cols=1,rows=2)
        g_subplotter.next(top_down_binary_image, 'bin')
        g_subplotter.next(histogram, 'hist', just_plot=True)
        g_subplotter.show()

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img_height/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = top_down_binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    #pdb.set_trace()

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_height - (window+1)*window_height
        win_y_high = img_height - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if 0:
            print("window:", window)
            print("  left, right x current:", leftx_current, rightx_current)
            print("  y low, high:", win_y_low, win_y_high)
            print("  x left low, high: ", win_xleft_low, win_xleft_high)
            print("  x right low, high:", win_xright_low, win_xright_high)
            print("  # pts left, right:", len(good_left_inds), len(good_right_inds))
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_height-1, img_height )
    left_fit = (0,1,0)
    right_fit = (0,1,0)

    # Fit a second order polynomial to each
    if len(leftx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        print("ERROR: Could not find left lane points")
        g_error_frames = g_error_frames + 1

    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        print("ERROR: Could not find right lane points")
        g_error_frames = g_error_frames + 1

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if g_debug_internal:
        g_subplotter.setup(cols=2,rows=2)
        g_subplotter.next(orig_image, 'orig')
        g_subplotter.next(top_down_binary_image, 'top_down')
        plt.plot(histogram)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 255, 255]
        #plt.imshow(out_img)
        g_subplotter.next(out_img, 'out')
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        #plt.show()

    if g_debug_internal: 
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((top_down_binary_image, top_down_binary_image, top_down_binary_image))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                      ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                      ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        #plt.imshow(result)
        g_subplotter.next(result, 'result')
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    # 6. Determine the curvature of the lane and vehicle position with respect to center.

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = (0,1,0)
    right_fit_cr = (0,1,0)
    if len(leftx) > 0:
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    if len(rightx) > 0:
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    if g_debug_internal: 
        plt.title(str(left_curverad) + "m " + str(right_curverad) + "m")

    # From project spec:
    # "You can assume the camera is mounted at the center of the car, such that
    # the lane center is the midpoint at the bottom of the image between the
    # two lines you've detected. The offset of the lane center from the center
    # of the image (converted from pixels to meters) is your distance from the
    # center of the lane."
    camera_center_x = img_width // 2
    left_lane_bottom_x = np.polyval(left_fit, y_eval)
    right_lane_bottom_x = np.polyval(right_fit, y_eval)
    center_lane_bottom_x = (right_lane_bottom_x - left_lane_bottom_x)/2 + left_lane_bottom_x
    center_offset_x = camera_center_x - center_lane_bottom_x
    center_offset_m = center_offset_x * xm_per_pix
    center_direction = "left"
    if center_offset_x > 0:
        center_direction = "right"
    if 0:
        print("left, center, right x:", left_lane_bottom_x, center_lane_bottom_x, right_lane_bottom_x)
        print("camera at x:", camera_center_x, "which is ", center_offset_x, "pixels", center_direction, "from center of lane")

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(top_down_binary_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # 7. Warp the detected lane boundaries back onto the original image.
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (orig_image.shape[1], orig_image.shape[0])) 

    # Combine the result with the original image
    # 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    result = cv2.addWeighted(orig_image, 1, newwarp, 0.3, 0)
    text = "Lane curvature: {:3.2f}m {:3.2f}m".format(left_curverad, right_curverad)
    cv2.putText(result, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)
    text = "Car is {:3.2f}m {} from center of lane".format(np.absolute(center_offset_m), center_direction)
    cv2.putText(result, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)

    if g_error_frames > 0:
        text = "Error frames: {}".format(g_error_frames)
        cv2.putText(result, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)

    if g_debug_internal: 
        plt.subplot(2,2,1)
        plt.imshow(result)
        g_subplotter.show()

    return result


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

    return window_centroids

def find_lane_lines_sliding_window(orig_image, top_down_binary_image):
    # Read in a thresholded image
    warped = top_down_binary_image
    # window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching

    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

# If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
                l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
                r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
     
# If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

# Display the final results
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()
    
def process_image_file(image_filename):
    if args.verbose:
        print(image_filename)

    image = mpimg.imread(image_filename) # reads in as RGB
    process_image(image)

def process_image(image):
    image = g_image_undistorter.undistort(image)

    #plot_colors(image)
    #return None

    binary_image = create_binary_image(image)

    top_down_binary_image, Minv = create_top_down_image(image, binary_image)
    #return

    final_image = find_lane_lines_basic(image, top_down_binary_image, Minv)
    #find_lane_lines_sliding_window(image, top_down_binary_image)
    return final_image

# 3. Use color transforms, gradients, etc., to create a thresholded binary image.
def create_binary_image(image):

    r_channel = image[:,:,0]
    #g_channel = image[:,:,1]
    #b_channel = image[:,:,2]

    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    #h_channel = hls[:,:,0]
    #l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    #gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(5,100))
    #grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(5, 100))
    #mag = mag_thresh(gray, sobel_kernel=ksize, thresh=(30, 100))
    #dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3)) # 0,np.pi/2 = full range

    #g_subplotter.setup(cols=2,rows=3)
    #g_subplotter.next(image, 'orig')
    #g_subplotter.next(gray, 'gray')
    #g_subplotter.next(gradx, 'grad-x')
    #g_subplotter.next(grady, 'grad-y')
    #g_subplotter.next(mag, 'mag')
    #g_subplotter.next(dir_binary, 'dir')
    #g_subplotter.show()

    if 0: # gradx,y
        #grad1 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(5,100))
        #grad2 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(25,100)) # winner
        #grad3 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(30,100))
        #grad4 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(50,100))
        #grad5 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(75,100))

        grad1 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(25,100))
        grad2 = abs_sobel_thresh(r_channel, orient='x', sobel_kernel=ksize, thresh=(25,100))
        grad3 = abs_sobel_thresh(r_channel, orient='x', sobel_kernel=ksize, thresh=(30,100))
        grad4 = abs_sobel_thresh(r_channel, orient='x', sobel_kernel=ksize, thresh=(50,100)) # winner
        # 50,100 is the winner because it's low noise and still picks up white lines well
        grad5 = abs_sobel_thresh(r_channel, orient='x', sobel_kernel=ksize, thresh=(75,100))

        #grad1 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(5,100))
        #grad2 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(10,100))
        #grad3 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(15,100))
        #grad4 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(20,100))
        #grad5 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(25,100)) # winner

        # 'x' is better than 'y'
        #grad1 = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=ksize, thresh=(5,100))
        #grad2 = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=ksize, thresh=(25,100))
        #grad3 = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=ksize, thresh=(30,100))
        #grad4 = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=ksize, thresh=(50,100))
        #grad4 = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=ksize, thresh=(75,100))

        g_subplotter.setup(cols=2,rows=3)
        g_subplotter.next(image, 'orig')
        g_subplotter.next(grad1, 'S_channel grad')
        g_subplotter.next(grad2, 'grad2')
        g_subplotter.next(grad3, 'grad3')
        g_subplotter.next(grad4, 'grad4')
        g_subplotter.next(grad5, 'grad5')
        g_subplotter.show()

    if 0: # direction
        dir_s1 = dir_threshold(s_channel, sobel_kernel=ksize, thresh=(-0.000001, -1.0))
        dir_s2 = dir_threshold(s_channel, sobel_kernel=ksize, thresh=(-0.05, 0.05))
        dir_s3 = dir_threshold(s_channel, sobel_kernel=ksize, thresh=(-0.1, 0.1))

        g_subplotter.setup(cols=2,rows=2)
        g_subplotter.next(image, 'orig')
        #g_subplotter.next(s_channel, 'HLS_X')
        g_subplotter.next(dir_s1, 'dir_s1')
        g_subplotter.next(dir_s2, 'dir_s2')
        g_subplotter.next(dir_s3, 'dir_s3')
        g_subplotter.show()

    if 0: # mag
        #mag_grey = mag_thresh(gray, sobel_kernel=ksize, thresh=(30, 100))
        #mag_s = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(30, 100)) # winner
        #mag_r = mag_thresh(r_channel, sobel_kernel=ksize, thresh=(30, 100))

        #mag_s1 = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(30, 100)) # winner
        #mag_s2 = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(50, 100))
        #mag_s3 = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(75, 100))
        #mag_s4 = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(90, 100))

        mag_s1 = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(30, 100)) # winner not much diff
        mag_s2 = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(30, 125))
        mag_s3 = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(30, 150))
        mag_s4 = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(30, 175))

        g_subplotter.setup(cols=2,rows=3)
        g_subplotter.next(image, 'orig')
        g_subplotter.next(s_channel, 's_channel')
        g_subplotter.next(mag_s1, 'mag-s1')
        g_subplotter.next(mag_s2, 'mag-s2')
        g_subplotter.next(mag_s3, 'mag-s3')
        g_subplotter.next(mag_s4, 'mag-s4')
        g_subplotter.show()

    if 0: # color thresh
        #thresh1 = color_threshold(s_channel, thresh=(170,255)) # winner
        #thresh2 = color_threshold(s_channel, thresh=(200,255))
        #thresh3 = color_threshold(s_channel, thresh=(215,255))
        #thresh4 = color_threshold(s_channel, thresh=(230,255))

        #thresh1 = color_threshold(s_channel, thresh=(170,255)) # winner
        #thresh2 = color_threshold(s_channel, thresh=(150,255))
        #thresh3 = color_threshold(s_channel, thresh=(125,255))
        #thresh4 = color_threshold(s_channel, thresh=(100,255)) # good in the shady spots

        #thresh1 = color_threshold(r_channel, thresh=(170,255)) # winner
        #thresh2 = color_threshold(r_channel, thresh=(150,255))
        #thresh3 = color_threshold(r_channel, thresh=(125,255))
        #thresh4 = color_threshold(r_channel, thresh=(100,255))

        thresh1 = color_threshold(r_channel, thresh=(170,255)) # winner
        thresh2 = color_threshold(r_channel, thresh=(190,255))
        thresh3 = color_threshold(r_channel, thresh=(210,255))
        thresh4 = color_threshold(r_channel, thresh=(230,255))

        g_subplotter.setup(cols=2,rows=3)
        g_subplotter.next(image, 'orig')
        #g_subplotter.next(s_channel, 's_channel')
        g_subplotter.next(r_channel, 'r_channel')
        g_subplotter.next(thresh1, '1')
        g_subplotter.next(thresh2, '2')
        g_subplotter.next(thresh3, '3')
        g_subplotter.next(thresh4, '4')
        g_subplotter.show()

    s_channel_thresh1 = color_threshold(s_channel, thresh=(170,255))
    r_gradx = abs_sobel_thresh(r_channel, orient='x', sobel_kernel=ksize, thresh=(50,100))
    gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(25,100))
    mag = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(30, 100))

    combined_binary = np.zeros_like(mag)
    combined_binary[(s_channel_thresh1 == 1) | (gradx == 1)] = 1
    combined_binary_r = np.zeros_like(mag)
    combined_binary_r[(s_channel_thresh1 == 1) | (gradx == 1) | (r_gradx == 1)] = 1 

    if g_debug_internal:
        g_subplotter.setup(cols=3,rows=3)
        g_subplotter.next(image, 'orig')
        g_subplotter.next(s_channel, 's_channel')
        g_subplotter.next(s_channel_thresh1, 's_thresh')
        g_subplotter.next(r_channel, 'r_channel')
        g_subplotter.next(r_gradx, 'r_gradx')
        g_subplotter.next(gradx, 'gradx')
        g_subplotter.next(mag, 'mag')
        g_subplotter.next(combined_binary, 's_thresh|gradx')
        g_subplotter.next(combined_binary_r, 's_thresh|gradx|r')
        g_subplotter.show()

    return combined_binary_r

def process_images(num, filenames=None):
    if args.verbose:
        print("Processing images")

    if filenames:
        image_filenames = glob.glob(filenames)
        #image_filenames = image_filenames + glob.glob('test_images/*.jpg')
    else:
        image_filenames = glob.glob('test_images/*.jpg')

    #print("files:", image_filenames)
    count = 0
    for image_filename in image_filenames:
        if (num != None) and (count >= num):
            break
        process_image_file(image_filename)
        count = count + 1

def process_video(filename):
    base_filename, file_ext = os.path.splitext(os.path.basename(filename))
    output_filename_no_ext = os.path.join(".", base_filename)
    output = output_filename_no_ext + "_processed" + file_ext

    # Note: use this to extract 1 jpg per second from a video:
    # -ss is the start time (19 seconds)
    # -t is the time to run the video (5 seconds)
    # -i is the input file
    # -r is the frame rate at which to grab jpgs (1.0 per second)
    # %4d adds an auto-incrementing file name
    # ffmpeg -ss 00:00:19 -t 00:00:05 -i project_video.mp4 -r 1.0 testout%4d.jpg
    clip = VideoFileClip(filename)
    #NOTE: fl_image() expects color images!!
    processed_clip = clip.fl_image(process_image)#.subclip(19,30) # seconds
    processed_clip.write_videofile(output, audio=False)

g_image_undistorter = ImageUndistorter()

def main():

    global g_debug_internal
    if args.verbose:
        print("being verbose")

    if args.calibrate:
        calibrate_camera()

    test_undistortion()
    sys.exit(1)

    if args.video:
        process_video("project_video.mp4")
        #process_video("challenge_video.mp4")
        #process_video("harder_challenge_video.mp4")
    else:
        g_debug_internal = True
        process_images(args.num_images, args.image_files)

if __name__ == "__main__":
    main()

