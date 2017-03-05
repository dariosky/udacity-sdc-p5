# Advanced Lane Finding Project

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
[camera-original]: output_images/camera_1_original.jpg "Camera original"
[camera-undistorted]: output_images/camera_2_undistorted.jpg "Camera undistorted"

[src]: output_images/test1_step1_original.jpg "Original"
[undist]: output_images/test1_step2_undistorted.jpg "Undistorted"
[birdeye]: output_images/test1_step3_birdeye.jpg "Road Transformed"
[binary]: output_images/test1_step4_binary.jpg "Binary Example"
[lines]: output_images/test1_step5_warped_detection.jpg "Fit Visual"
[output]: output_images/test1_step6_detection.jpg "Output"

[video-out]: video/project_video_output.mp4 "Video"

---
### General overview

The main class for dealing with the project is the `ProcessPipeline` class in `pipeline.py:40`.

It deals with the required functions: calibrating the camera, loading the images,
keep memory of previous detections, and receiving messages from the pipeline to display
proper messages.
There are some example usage functions, run_single, run_video and so on in the same file.

A Jupyter notebook `p4.ipynb` is also included, there are some experimental steps I used while choosing the hyper-parameters for the detection.


### Camera Calibration

The `ProcessPipeline` class has a method `ensure_calibrated` that check compute the trasformation matrix required to undistort the camera original frames, calling the
`camera_calibration` method.

The `linefinder.camera_calibration.get_calibration_points` method starts
receiving a list of camera images of the chessboard 9x6, (setting display=True parameter
  the output is shown on screen).
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

Original image             |  Undistored
:-------------------------:|:-------------------------:
![][camera-original]       |  ![][camera-undistorted]

The undistortion matrix is kept in the Pipeline for further usage.
To avoid recompute it, it's also serialized and stored on disk in the `calibration_points.pickle` file.


### Pipeline (single images)

An example of the pipeline usage for single image is made by calling `run_single(filename, show_steps=True, save_steps=False)`, given the filename we can choose to visualize all the steps and/or save them on the `output_images` folder.


#### 1. distortion-correction
Given the undistortion matrix, apply it via `c2.undistort` is straightforward, it's done in the `pipeline.ProcessPipeline#undistort` method.

Original | Undistorted
---------|------------
![][src] | ![][undist]


#### 2. Perspective transform

In the file `linefinder/birdeye.py` I start by choosing an "asphalt_box", that is a region on the street that will be warped to become a rectangle. It should be made of 4 points in the street that form a rectangle.

This rectangle will be mapped in another rectangle that is created by the `get_dstbox` that is a rectangle with the same X proportion but with stretched to fill the image height.

To choose the 4 points in the asphalt I plotted them on the undistorted image above (on a frame with straight lines).

The order should be coherent, I did (bottom left, top left, bottom right, top right).
Here are the chosen points:

|           | Source        | Destination   |
|: --------:|:-------------:|:-------------:|
|close left | 690, 256      | 719, 256      |
|far left   | 460, 580      | 0, 256        |
|close right| 690, 1058     | 719, 1058     |
|far right  | 460, 705      | 0, 1058       |

The distortion matrix and the inverted distortion matrix are computed and stored (to reuse them on each frame).

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Undistorted | Bird-eye-view
------------|------------
![][undist] | ![][birdeye]


#### 3. Line detection

The relevant code for the edge detection is in `linefinder/edges.py`.

I used a combination of three methods:
*  Threshold the gradient of Sobel X computed on the luminosity layer normalized
*  Threshold the saturation channel (it helps with yellow lines)
*  Threshold a normalized luminosity layer (it helps with white lines)

To adjust the threshold values I did lot of experiments with road images and with live camera input in open CV.

Here are the threshold levels I choosed, they works well recognizing lines and filtering out the noise.

| function   | Threshold filter |
|------------|------------------|
| SobelX     | [20, 100)        |
| Luminosity | [220, 255)       |
| Saturation | [160, 240)       |

Here's an example of my output for this step.  (note: this is not actually from one of the test images)

Bird-eye     | Binary
------------ |------------
![][birdeye] | ![][binary]


#### 4. Finding lane-line pixels and fit their positions with a polynomial

In the pipeline `pipeline.py:187` there is a call to the `get_line_points` function, the code for line detections is `linefinder/lines.py`.

`get_line_points` uses the `find_window_centroids` function that use a convolution of a rectangular window (so that in an image we have 30x8 windows) using a suggested_centers to reuse the previous frame data (if available).

Given the image it return an array of WINDOWS_Yx2 array with the points of the left and right lane. Those are then used in `get_left_right_lanes` to compute quadratic fit function.
(the terms of the polynomial is also preserved for the future frames, in case we miss a detection).

I am using in those function the `blinker` library to send messages to the Pipeline class, it's a convenient way to send messages across without having to pass callbacks around.

Finally the `line_on_the_road` uses the polynomial function to highlight the lane area in the warped image, and uses the inversed distortion matrix to return in the undistorted format so to produce the final output.

I also left the `get_convoluted_lines`, that uses the convoluted centroid to draw the pixels of lanes with a grayscale version of the birdeye, this is quite useful in debug mode.

Here some images of the intermediate steps

Bird-eye     | Binary      | Lane detected |
------------ |-------------|---------------|
![][birdeye] | ![][binary] | ![][lines]

#### Calculated the radius of curvature

The radius of the lanes is computed in `linefinder.lines.radius_in_meters` it uses the proportions of meters/pixel given in the class. It produces coherent results.
Radius is calculated on the last point of the considered detected lines.

The distance from center is calculated in `pipeline.py:203`, it find the middle position from the closest point of the detected lanes and compute the shift from the center of the image (whete the camera is placed).

The output of both functions is overlayed in the output image below.

![][output]


#### Messages and stability

When used in sequences or on a video, the pipeline use previous states to improve stability.

In the top right a message is shown when we use the "blind scan" to detect the lane position (without using previous known positions) or a "can't find left/right lane" when one or both the lanes cannot be found with enough certainty.

In this case I first try a blind scan, and if even then I'm below a certain threshold I reuse the previous polynomial to reuse the lanes of the previous frame.

---

### Pipeline (video)

A video output of the project video is in
Here is a [link to my video result](video/project_video_output.mp4)

<iframe width="560" height="315" src="https://www.youtube.com/embed/wQrSSEXhPIA" frameborder="0" allowfullscreen></iframe>
---

### Discussion

I enjoyed a lot, playing with OpenCV. It is marvelous, and following the guideline in the course I reached decent results on the project video quickly.

I spent some time trying to stabilize the video, the major problem was that
when only part of the lane lines were detected the find_window_centroids function
was considering the leftmost part of the window as a valid center (as a consequence of the argmax call). It took me a while to understand what was happening, so I started debugging the code and adding utility functions (part of them are still there, they are activated setting DEBUG=True on lines.py).

Pipeline can be made more robust, considering the expected lane size, so that whenever I have a confident lane, the other can be supposed even when not detected.

Also the line position, when detected, can be stabilized with previous detections.
I didn't implementated because the result is already quite nice, and without stabilitation the response to sudden line changes are detected quickly.

Looking forward for adding Machine Learning techniques to the pipeline both for detecting other cars and to output other parameters.
