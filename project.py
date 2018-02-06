#!/usr/bin/env python3

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import statistics
import os
import math

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import argparse

# The goals / steps of this project are the following:
# 
# 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# 2. Apply a distortion correction to raw images.
# 3. Use color transforms, gradients, etc., to create a thresholded binary image.
# 4. Apply a perspective transform to rectify binary image ("birds-eye view").
# 5. Detect lane pixels and fit to find the lane boundary.
# 6. Determine the curvature of the lane and vehicle position with respect to center.
# 7. Warp the detected lane boundaries back onto the original image.
# 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 

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
    print("Determining camera calibration and distortion coefficients")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", '--verbose', action='store_true', help="be verbose")
    parser.add_argument("-c", '--calibrate', action='store_true', help="determine camera calibration")
    args = parser.parse_args()
    if args.verbose:
        print("being verbose")

    if args.calibrate:
        calibrate_camera()
        

if __name__ == "__main__":
    main()

