** Vehicle Detection Project **
===============================

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[carnotcar]: output_images/car_noncar.png
[hog]: output_images/hog_2976.png
[spatial&hist]: output_images/spatial_col_features.png
[test6vt]: output_images/test6_step7_vehicle_tracking.jpg

---
## General overview

The code for this project is mainly in the `vehicle_detection` package.

A standalone example of usage is the `heat.py` file, where the trained
SVN is used to build the bounding boxes classified as cars, and heat-map is constructed.

The `train.py` is used to do the initial training for the classifier, it
will be described in detail below.

The rest of the project is a fusion with the advanced lane detection project
the existing pipeline has been merged with the steps required for vehicle
detection. The main entry point is the `pipeline.py` file.


## Training

For the training phase (`train.py`) I used the full dataset provided,
a `get_car_dataset` function (`train.py:30`) get the list of filepaths from the
dataset it then keep a `dump` that will be saved to file (I used the `joblib`
library for this, as it's more efficient than pickle for big numpy arrays).

For all the images, I extracted 3 different features: HOG, spatial (a color resized image) and a bucketed-histogram of colors. The details below.

The function for extracting the features is `vehicle_detection.combine.extract_features`.
I kept the parameters I used for training as a default, but they're saved in the dump file,
it's important that the same parameters are used both in training and in the prediction/test.

I made various experiment combining various feature-sizes and enabling/disabling the various features. I ended up with this final parameters, with these I got 99% accuracy in the classifier.

### Histogram of Oriented Gradients (HOG)

I computed the HOG for the dataset images in all three channels in the YCrCb colorspace.
Using the 3 channels gave me better results than using a single one (the Y one was a good candidate).  
The code for this step is contained in the `vehicle_detection/combine.py:52`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![][carnotcar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

Here is an example using the final HOG parameters (here only one channel), in the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

![][hog]

#### 2. Other features: bin_spatial & color_hist

I found that adding also the binary `vehicle_detection.color_features.bin_spatial` and `vehicle_detection.color_features.color_hist` helped increasing the accuracy.
I used a 32x32 resize for the binned spatial and 32 buckets for the color histogram procedure. Here an example:

![][spatial&hist]

#### 3. Choosing the HOG parameters

I tried various combinations of parameters, I was relying on the accuracy level
to decide if a change was beneficial or not. Due to the random way of picking the test-set the accuracy may differ across training sessions, but with the final values I got a good 99%.

#### 4. Training the Linear SVM classifier

The training phase happens `vehicle_detection.train.train`, it merge the 3 channels HOG with the bin_spatial and the color_hist features. The various features are normalized
to have zero mean and deviation 1, by using `StandardScaler` preprocessor.

The same scaler, is saved on the dump for subsequent reusage.

I trained a linear SVM using the fit method, and tested the accuracy with a test sample with 20% the size of the original dataset.

### Sliding Window Search

The sliding windows search is part of the `vehicle_detection.combine.find_car_boxes` method.
It takes all the parameters given to the train method above, and proceeds doing a sliding window lookup, over the picture, computing the HOG function only once, in the whole image, and reusing its slices to get the window features.

The procedure returns a list of rectangle positions where it found positive car matches.

I found that a refinement phase to remove some false_positive was beneficial here,
so after training the SVM the first time, I run the `find_car_boxes` with the `save_positive` parameter, it then saves all the positive boxes occurred in the left part of the screen (where the number of detections are quite low). I filtered them manually to remove the real car detections and did a refinement steps on the training (they are ~400 images with false positives). The train method, process them as false positive, scanning the path passed in the `false_positive_path` parameter.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As said before, I got the best result with 3 channels HOG, plus spatially binned color and histograms of color in the feature vector, and a final refinement process marking as noncars all the false detections.  Here is an example image:

![][test6vt]
---

### Video Implementation

#### The final video output for the project.

Here is a [link to my video result](video/project_video_output.mp4)

<iframe width="560" height="315" src="https://www.youtube.com/embed/sY-KxY-EGVs" frameborder="0" allowfullscreen></iframe>


#### Video pipeline discussion

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here is a [link to the test video with visible heatmaps](video/test_video_output.mp4)

<iframe width="560" height="315" src="https://www.youtube.com/embed/Tf4HfcY-Hdk" frameborder="0" allowfullscreen></iframe>

#### Video pipeline stabilization

I found the part of stabilizing the pipeline for the video processing one of the
most interesting parts (`pipeline.ProcessPipeline#detect_and_heat`).
In every frame I detect the heatmap, and discard the cells with a single detection (the `SINGLE_FRAME_THRESHOLD = 1` parameter), I add this heatmap with a rolling heatmap made in the previous frames, capping the value to a maximum of `MAXIMUM_HEAT = 15`.
I then consider as valid detections the contiguous label using a threshold of `ROLLING_HEAT_THRESHOLD = 5`. Finally, at every frame, the rolling heatmap is "cooled down" with `HEAT_GENERATION_DECAY = 1`.
These parameters gave me a good result, they help removing the spurious false detections and react quite fast to detected car movements.

---

### Discussion

I found the computer vision part of this course extremely interesting.
In this project the feature extractions with various techniques and the rolling windows
search are really powerful.
For time constraint I still didn't spent much time on the challenge videos,
but surely there are many improvements possible for the future.

The classifier, is having a good accuracy, but I saw that there are still spurious artifacts on the lanes (mostly filtered out via a threshold), but I would like to test a CNN, I think the neural network should perform better in this case.  
I will try also one of the big datasets available, the ones provided by [Udacity](https://github.com/udacity/self-driving-car/tree/master/annotations)
seems big enough and have additional labeling.

I found this last project useful, the rolling-window technique is useful in a plethora of cases, and it's a good final project for this interesting Nanodegree.
