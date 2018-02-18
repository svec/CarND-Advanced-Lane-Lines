## Advanced Lane Finding Writeup

**Advanced Lane Finding Project**

*Submitted by Chris Svec, February 2018*

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[distortion_image]:   ./output_images/calibration2-undistorted.png "Distorted and undistorted images"
[distortion_test]: ./output_images/test1-undistorted.jpg "Undistorted test image"
[binary_image]: ./output_images/writeup-binary.png "Thresholded binary image"
[masked_image]: ./output_images/writeup-mask.png "Masked image"
[transformed_image]: ./output_images/writeup-trans.png "Transformed image"
[lane_image]: ./output_images/writeup-lane-finding.png "Finding lanes"
[final_image]: ./output_images/writeup-final.png "Final output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

All code for this project is in the [project.py](project.py) Python 3 file.

The code for this step is in the calibrate_camera() function starting at line 136. 

I started with the code from the lectures and modified it to suit this project.

Camera calibration started with several 6x9 checkerboard images. These images are
of the same physical checkerboard taken from different angles and distances.
The checkerboard images are fed into the OpenCV findChessboardCorners() function,
which locates each checkerboard corner in each checkerboard image.

The corner locations and corresponding 6x9 grid locations are fed into the
OpenCV calibrateCamera() function, which returns the distortion coefficients
needed to remote camera distortion from images taken with that camera.

The OpenCV undistort() function takes these distortion coefficients and an
original distorted image and returns an undistorted image.

Here is an original, distorted image and the undistorted version:
![alt text][distortion_image]

You can see the curvature on the top of the original, distorted image on the
left. The curve is gone in the undistorted image on the right: it has a flat top.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied distortion correction to each image on line 824. Here's an example of an
undistorted version of one of the Udacity supplied test images:

![alt text][distortion_test]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used the create_binary_image() function to create a thresholded binary image.

I created the thresholded binary image by combining:
1. The 'x' gradient (Sobel) with threshold (25,100) applied to the LAB L channel data with threshold (200,255)
2. The 'x' gradient (Sobel) with threshold (25,100) applied to the LAB B channel data with threshold (143,255)
3. The 'x' gradient (Sobel) with threshold (50,100) applied to the RGB R channel data with no threshold.

The final image is created in lines 1008-1022.

Here's an example of an original image and its binary thresholded output:

![alt text][binary_image]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform happens in the function create_top_down_image().

Before transforming the perspective, I masked off the camera view directly in
front of the car so that only it is considered in the rest of the pipeline. This
is done with the region_of_interest() call at line 417.

The actual masking was done on the binary thresholded image, but here is what the
mask looks like on the original image (it's easier to tell what's masked in this
color image):

![alt text][masked_image]

The perspective transform happens starting at line 427. I created a quadrilateral
for the original image, and mapped that onto a rectangle in the transformed image.

The quadrilateral applied to the first image was determined by trial and error,
as suggested by the Udacity lectures.  I started with an image of a straight
road. I displayed the quadrilateral on the first image and the rectangle on the
transformed image and tweaked the quadrilateral size and shape until the lane
lines in the transformed image were vertical and parallel.

The perspective transform was done with the OpenCV getPerspectiveTransform() and
warpPerspective() functions.

This example shows the quadrilateral drawn in red lines on the left image, and
the rectangle drawn in red lines on the right image. The vertical green lines
on the right image were drawn at the lane edges to visually verify that the
lanes were vertical:

![alt text][transformed_image]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify lane lines and fit them to a polynomial, I boldly ~~stole~~ used the
Udacity lecture's suggested code in my find_lane_lines_basic() function.

This code starts with the bottom half of transformed binary image. It finds the
window with the most non-zero pixels on the left half and right half of the
image and uses those as the starting points for the left lane and right lane.

It then moves the window up the image looking for more non-zero pixels in the
windows. It shifts the window left or right based on the average position of the
non-zero pixels in the previous window. All the non-zero points within the left
windows are used for the left lane points, and all the non-zero points within
right windows are used for the right lane points.

This picture shows the windows in the green, the left lane points in red and the
right lane points in white. Points that fall outside of any of the windows are
not used.

![alt text][lane_image]

The numpy polyfit() function is used to fit the left and right lane points to a
second order polynomial, which are drawn as yellow lines in that image.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature of the lanes and the position of the
vehicle around using the algorithms described in lecture.

Basically I mapped the pixels to the real world distances using the estimates
provided in lecture, and as Will Smith says, I got mathy with it in lines
664-679.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step starting at line 705. The polyfit lane lines were drawn
onto a blank image and filled in with green. This image was then perspective
transformed back to the original image orientation using the inverse perspective
matrix and the warpPerspective() function. The resulting image showed the lane
lines in the same perspective as the original image, which results in a final
image like this:

![alt text][final_image]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video_processed.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline has a hard time seeing white lines on lighter gray pavement. I
added the RGB Red channel to improve the visibility of these lines, but it could
be improved.

Varying colors in the pavement between lane lines also slightly confuses my
pipeline: perhaps some pixel masking could help, or some better color filtering
methods.

After changing to the LAB L + LAB B + RGB R channels (from HLS S + RGB R) the lane finding works much better, but still has a few frames where the pipeline gets slightly confused on the left lane when the pavement vs line contrast is low. The pipeline finds most of the lane correctly; it diverges at its furthest point away from the car.
