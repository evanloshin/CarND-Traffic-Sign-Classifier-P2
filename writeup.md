# **Traffic Sign Recognition**

## By: Evan Loshin
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./writeup-images/labels-hist-normed.jpeg
[image4]: ./writeup-images/example-images.jpeg
[image5]: ./writeup-images/equalization.jpeg
[image6]: ./writeup-images/preprocessing.jpeg
[image7]: ./writeup-images/augmentation-hists.jpeg
[image8]: ./writeup-images/my-images.jpeg
[image9]: ./writeup-images/my-images-results.jpeg
[image10]: ./writeup-images/my-images-scatter.jpeg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Follow the link to view my full [project submission](https://github.com/evanloshin/CarND-Traffic-Sign-Classifier-P2) on Github

My project was constructed in this [Jupyter Notebook](https://github.com/evanloshin/CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)

Here is an [html version](https://github.com/evanloshin/CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 images
* The size of the validation set is 4,410 images
* The size of test set is 12,630 images
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

Here are some example images from the dataset:

![image4]

#### 2. Include an exploratory visualization of the dataset.

The distribution of labels for the three subsets of images is pictured below.

![image3]

The *Train* dataset is deficient in images with labels in the 25-30 range. I expect the model to struggle with classifying images in this range. Additionally, the accuracy will be biased since the *Validation* and *Test* subsets will test the model primarily on images outside this range.

To emphasize this point, I used the Collections library to generate statistics on the distribution of labels for the *Train* subset:
* The average number of images per label is 809
* The largest number of images per label is 2010
* The smallest number of images per label is 180

A pretty large range!


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I convert the images to grayscale to simplify the computational requirements of the neural network. Here's an example of a traffic sign image before and after grayscaling:

![image2]

As a second step, I equalize the image to amplify features because many images in the dataset are pretty dark. After some experimentation, this step had negligible  benefit. Here's an example before and after equalization:

![image5]

I also decided to generate additional data because of the deficiencies mentioned in section 2 of my writeup. To add more data to the the data set, I rotated images by random angles (-15d to +15d) using the *rotate* method in the SciPy Python library. I needed to identify deficient label categories and iterate the *Train* dataset to supplement only those categories. Take a look at the Jupyter Notebook to see how I implemented this.

Here is an example of an original image and an augmented image:

![image6]

Here are the distributions of images before and after augmentation:

![image7]

Last, I normalize the pixel values to help the optimizer find the solution space faster and reduce the likelihood of falling into a local optimum during gradient decent. I use a tensor node to easily accomplish normalization. See the table in the next section on model architecture.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model architecture consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|	-								|
| Max pooling 2x2	      	| 2x2 stride, valid padding, outputs 14x14x6 |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU          | -               |
| Max Pooling 2x2     | 2x2 stride, valid padding, outputs 5x5x16   |
| Flatten       | outputs 400     |
| Fully connected		| outputs 120     	|
| RELU          | -               |
| Fully Connected     | outputs 84       |
| RELU          | -               |
| Fully Connected     | outputs 43      |
| Softmax				| -        									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 128 and learning rate of 0.0008 for 25 epochs. I found these parameters using trial and error to maximize the validate accuracy.

The model uses the Adam Optimizer introduced in lab.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **99.9%**
* validation set accuracy of **95.3%**
* test set accuracy of **93.3%**

I took an iterative approach to developing my model. This way the results of each new modification were measured individually. I used the LeNet5 architecture with grayscale images. This architecture was developed for image classification and we studied it in lab, so it was a logical choice.

To start, my model plateaued around 90% validation accuracy after just a few epochs. My next priority was tuning the hyperparameters to maximize the effectiveness of my initial architecture. After several trials with no clear improvement, I decided to keep the following initial parameters from lab:
* 25 epochs
* batch size of 125
* learning rate of 0.001

Next, I implemented equalization in my image pre-processing pipeline. The result was a marginal improvement. Then, I experimented with the LeNet architecture by adding another max pooling layer and changing the dimensions of the strides in my convolution layers. These experiments hurt the accuracy so I reverted them.

These mostly unsuccessful modifications sent me looking for opportunities external to the architecture. The final logical approach was augmenting the dataset. I added 10,214 rotated images to the *Train* subset. These additions brought the validation accuracy up to 93%.

Determined to maximize the accuracy but out of ideas, I wondered if the plateau in my initial iteration could've been overcome. I decided to increase the number of epochs to 25 and let the model run unattended for awhile. Sure enough, that bought me an extra 2%. Perhaps even higher accuracy is still possible. Although, the training accuracy is near 100% so any more epochs would probably just reiterate over-fitting.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![image8]

The first image might be difficult to classify because the circle is elongated. The second image presents a challenge due to the sticker on the bottom of the arrow. The third image is slightly rotated. Finally, the last two images have distracting backgrounds.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![image9]

The model was able to correctly guess 4 out of 5 traffic signs, which gives an accuracy of 80%. This isn't as good as the accuracy on the *Test* dataset. However, the failure to classify the first image is no surprise. Not only is the circle extruded, the label falls into one of the most underrepresented categories in the *Train* dataset.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the top five softmax probabilities for each image:

![image10]

For the first image, the model is only slightly certain that this is the 120kph speed limit sign (probability of 0.4). Not surprisingly, it also predicts a relatively high likelihood (probability of 0.6) that this is a 20kph speed limit sign.

The model produces high probabilities (greater than 0.9) for the remaining four images. For each image, all the other probabilities nearly equal 0. This means the model is very confident in it's predictions.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I visualized feature maps for the first and second convolutional layers on the second of the five images from the section above. The maps in the first layer each include some combination of the sign's border and interior design. The maps in the second layer, however, are too pixelated to discern much of a pattern. I can barely make out somewhat of a pattern in one of the maps that looks like the arrows in the center of the sign. In a couple of the maps, I notice what looks like different edges of the circle.
