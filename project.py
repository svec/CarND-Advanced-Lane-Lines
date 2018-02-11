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

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

distortion_coeffs_pickle_file = "camera_cal/dist_pickle.p"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", '--verbose', action='store_true', help="be verbose")
parser.add_argument("-c", '--calibrate', action='store_true', help="determine camera calibration")
parser.add_argument("-n", '--num_images', type=int, help="number of images to process")
args = parser.parse_args()

# The goals / steps of this project are the following:
# 
# X 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# X 2. Apply a distortion correction to raw images.
# X 3. Use color transforms, gradients, etc., to create a thresholded binary image.
# _ 4. Apply a perspective transform to rectify binary image ("birds-eye view").
# _ 5. Detect lane pixels and fit to find the lane boundary.
# _ 6. Determine the curvature of the lane and vehicle position with respect to center.
# _ 7. Warp the detected lane boundaries back onto the original image.
# _ 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
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

    def next(self, image, title=None):
        if self.current == 0:
            print("ERROR: subplot next called before setup")
            sys.exit(1)

        if self.current > (self.cols * self.rows):
            print("ERROR: too many subplots for rows, cols:", self.rows, self.cols)
            sys.exit(1)

        plt.subplot(self.rows, self.cols, self.current)
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
    img = cv2.imread('test_images/test1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_undistorter = ImageUndistorter()
    dst = image_undistorter.undistort(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30) 
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

    if False:
        g_subplotter.setup(cols=3,rows=2)
        g_subplotter.next(rgb_image, 'original')
        #g_subplotter.next(gray, 'gray')
        #g_subplotter.next(gray, 'gray2')
        g_subplotter.next(r_channel, 'R')
        #g_subplotter.next(g_channel, 'G')
        #g_subplotter.next(b_channel, 'B')
        #g_subplotter.next(hls_h_channel, 'HLS_H')
        g_subplotter.next(hls_l_channel, 'HLS_L')
        g_subplotter.next(hls_s_channel, 'HLS_S')
        #g_subplotter.next(hsv_h_channel, 'HSV_H')
        #g_subplotter.next(hsv_s_channel, 'HSV_S')
        g_subplotter.next(hsv_v_channel, 'HSV_V')
        g_subplotter.show()

    if True:
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

def color_threshold(image, thresh=(0,255)):
    s_binary = np.zeros_like(image)
    s_binary[(image >= thresh[0]) & (image <= thresh[1])] = 1
    return s_binary

# _ 4. Apply a perspective transform to rectify binary image ("birds-eye view").
def create_top_down_image(orig_image, binary_image):
    # Choose offset from image corners to plot detected corners
    # This should be chosen to present the result at the proper aspect ratio
    # My choice of 100 pixels is not exact, but close enough for our purpose here
    offset = 100 # offset for dst points
    # Grab the image shape
    img_size = (binary_image.shape[1], binary_image.shape[0])

    # Source images are 1280 wide by 720 high.

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
    if 1:
        lines_on_orig_image = np.copy(orig_image)
        cv2.line(lines_on_orig_image, tuple(src[0]), tuple(src[1]), color=[255,0,0], thickness=1)
        cv2.line(lines_on_orig_image, tuple(src[1]), tuple(src[2]), color=[255,0,0], thickness=1)
        cv2.line(lines_on_orig_image, tuple(src[2]), tuple(src[3]), color=[255,0,0], thickness=1)
        cv2.line(lines_on_orig_image, tuple(src[3]), tuple(src[0]), color=[255,0,0], thickness=1)
        g_subplotter.setup(cols=2,rows=2)
        g_subplotter.next(lines_on_orig_image, 'orig')
        #g_subplotter.show()

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
    # Warp the image using OpenCV warpPerspective()
    top_down = cv2.warpPerspective(binary_image, M, img_size)

    top_down_color = cv2.warpPerspective(orig_image, M, img_size)

    if 1:
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

def process_image(image_filename):
    if args.verbose:
        print(image_filename)

    image = mpimg.imread(image_filename) # reads in as RGB
    image = g_image_undistorter.undistort(image)

    binary_image = create_binary_image(image)

    top_down_image = create_top_down_image(image, binary_image)

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
        grad1 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(5,100))
        grad2 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(25,100)) # winner
        grad3 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(30,100))
        grad4 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(50,100))
        grad5 = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(75,100))

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
        #g_subplotter.next(s_channel, 'HLS_X')
        g_subplotter.next(grad1, 'gradx1')
        g_subplotter.next(grad2, 'grad2')
        g_subplotter.next(grad3, 'grad3')
        g_subplotter.next(grad4, 'grad4')
        g_subplotter.next(grad4, 'grad5')
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
        #s_channel_thresh1 = color_threshold(s_channel, thresh=(170,255)) # winner
        #s_channel_thresh2 = color_threshold(s_channel, thresh=(200,255))
        #s_channel_thresh3 = color_threshold(s_channel, thresh=(215,255))
        #s_channel_thresh4 = color_threshold(s_channel, thresh=(230,255))

        s_channel_thresh1 = color_threshold(s_channel, thresh=(170,255)) # winner
        s_channel_thresh2 = color_threshold(s_channel, thresh=(150,255))
        s_channel_thresh3 = color_threshold(s_channel, thresh=(125,255))
        s_channel_thresh4 = color_threshold(s_channel, thresh=(100,255)) # good in the shady spots

        g_subplotter.setup(cols=2,rows=3)
        g_subplotter.next(image, 'orig')
        g_subplotter.next(s_channel, 's_channel')
        g_subplotter.next(s_channel_thresh1, 's_1')
        g_subplotter.next(s_channel_thresh2, 's_2')
        g_subplotter.next(s_channel_thresh3, 's_3')
        g_subplotter.next(s_channel_thresh4, 's_4')
        g_subplotter.show()

    s_channel_thresh1 = color_threshold(s_channel, thresh=(170,255))
    gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(25,100))
    mag = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(30, 100))

    combined_binary = np.zeros_like(mag)
    combined_binary[(s_channel_thresh1 == 1) | (gradx == 1)] = 1

    if 0:
        g_subplotter.setup(cols=2,rows=3)
        g_subplotter.next(image, 'orig')
        g_subplotter.next(s_channel, 's_channel')
        g_subplotter.next(s_channel_thresh1, 's_thresh')
        g_subplotter.next(gradx, 'gradx')
        g_subplotter.next(mag, 'mag')
        g_subplotter.next(combined_binary, 's_thresh|gradx')
        g_subplotter.show()

    return combined_binary

def process_images(num):
    if args.verbose:
        print("Processing images")

    image_filenames = glob.glob('test_images/*.jpg')

    count = 0
    for image_filename in image_filenames:
        if (num != None) and (count >= num):
            break
        process_image(image_filename)
        count = count + 1

g_image_undistorter = ImageUndistorter()

def main():

    if args.verbose:
        print("being verbose")

    if args.calibrate:
        calibrate_camera()

    #test_undistortion()

    process_images(args.num_images)

if __name__ == "__main__":
    main()

