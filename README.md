# CarND-Vehicle-Detection
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car_not_car.png
[image2]: ./images/HOG_example.jpg
[image3]: ./images/sliding_windows.png
[image4]: ./images/sliding_windows2.png
[image5]: ./images/bboxes_and_heat.png
[image6]: ./images/labels_map.png
[image7]: ./images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4 - 8 code cells of the IPython notebook called `main.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Using `YCrCb` color space and all hog channels appeared to give the best results. In addition, using all spatial, histogram and hog features also gave good results.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using linear SVC, by extracting the car and non-car features, normalising it before training. The data was split into 80% training and 20% testing. The code is in cell 9 of the IPython notebook called `main.ipynb`.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window was implemented in the function `find_cars()` in cell 12. The scale was using 1.0, 1.5, and 2.0, of window size 64x64, as different size cars and features appear in the frames. The overlap of 75% was chosen as it's a good amount of overlap without taking a long time to process. The scales and overlap were tested and gave good results in the pipeline.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To reduce false positives, I only performed sliding window in parts of the image from y pixels 350 to 656. This will ignore the top of the image where there's no vehicles.

I created 64x64 windows, scaled it by 1.0, 1.5, and 2.0, and slide the windows over different parts of the image. I recorded the positions of the positive detections for each scale, combined them, created a heatmap, then used a threshold of 4.0 to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.

Furthermore, I averaged the heapmaps over the last 15 frames.
I assumed each blob corresponded to a vehicle and constructed bounding boxes to cover the area of each blob detected. The code is in cell 19 of `main.ipynb`.

 
I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap, averaged it over the last 10 frames and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle or group of vehicles.  I constructed bounding boxes to cover the area of each blob detected. The code is in cell 19 of `main.ipynb`.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One problem I faced is that when using `RGB` color space, there's a lot of false negatives. Changing to `YCrCb` color space helped. 

The other problem was that using sliding windows over the whole window was producing a lot of false postives, and is very slow, so I only used the bottom half of the image (y pixels 350 to 656), and for each scale, only parts of the sub-image. This worked well to reduce the false positives, as there can't be any cars in the top part of the image, and small windows in the lower part of the sub-image produce false positives too, as vehicles in lower parts of the image tend to appear larger in size. This also improved performance, as there's less windows to search through.

The pipeline may fail where there's cars that are further away from the camera, and appear smaller in size. The pipeline may also fail where non cars are detected as cars, as there's not enough training data. Furthermore, the pipeline may fail to detect motorbikes, as there may not be motorbikes in the training data for svc to learn what motorbike features look like.

If I was going to pursue this project further, I would use transfer learning to improve the accuracy of the detection, collect more data (e.g. use the Udacity data), augment the images better and split the training images better using time series data.
