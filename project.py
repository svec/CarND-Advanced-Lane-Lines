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
# _ 3. Use color transforms, gradients, etc., to create a thresholded binary image.
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
        #   HLS_S is the only one that works
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

# 3. Use color transforms, gradients, etc., to create a thresholded binary image.
def process_image(image_filename):
    if args.verbose:
        print(image_filename)

    image = mpimg.imread(image_filename) # reads in as RGB
    undistorted_image = g_image_undistorter.undistort(image)

    plot_colors(undistorted_image)

    r_channel = undistorted_image[:,:,0]
    g_channel = undistorted_image[:,:,1]
    b_channel = undistorted_image[:,:,2]

    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    # Plotting thresholded images
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    #ax1.set_title('Stacked thresholds')
    #ax1.imshow(color_binary)

    #ax2.set_title('Combined S channel and gradient thresholds')
    #ax2.imshow(combined_binary, cmap='gray')
    #plt.show()

    if False and args.verbose:
        g_subplotter.setup(cols=2,rows=2)
        g_subplotter.next(undistorted_image, 'original')
        g_subplotter.next(sxbinary, 'sobel binary')
        g_subplotter.next(s_binary, 'S channel binary')
        g_subplotter.next(color_binary, 'color binary')
        #g_subplotter.next(color_binary)
        #g_subplotter.next(combined_binary)
        g_subplotter.show()
    


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

