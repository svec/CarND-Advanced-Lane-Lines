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

class ImageUndistorter:
# 2. Apply a distortion correction to raw images.
    def __init__(self):
        self.init = False
        self.mtx = None
        self.dist = None

    def undistort(self, image):
        if self.init == False:
            print("unpickling")
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

# 3. Use color transforms, gradients, etc., to create a thresholded binary image.
def process_image(image_filename):
    if args.verbose:
        print(image_filename)
    image = cv2.imread(image_filename)
    image = image_undistorter.undistort(image)


def process_images():
    if args.verbose:
        print("Processing images")

    image_filenames = glob.glob('test_images/*.jpg')

    for image_filename in image_filenames:
        process_image(image_filename)
  
image_undistorter = ImageUndistorter()

def main():

    if args.verbose:
        print("being verbose")

    if args.calibrate:
        calibrate_camera()

    #test_undistortion()

    process_images()

if __name__ == "__main__":
    main()

