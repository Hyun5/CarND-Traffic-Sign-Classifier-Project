# **Traffic Sign Recognition** 

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

[image0]: ./examples/Sample_traffic_signs.JPG "Visualization"
[image1]: ./examples/Visualization.JPG "Visualization"
[image2]: ./examples/Grayscale.jpg "Grayscaling"
[image3]: ./examples/Normalized.JPG "Normalized"
[image4]: ./examples/Zeichen_13.png "Traffic Sign 1"
[image5]: ./examples/Zeichen_14.png "Traffic Sign 2"
[image6]: ./examples/Zeichen_17.png "Traffic Sign 3"
[image7]: ./examples/Zeichen_3.png "Traffic Sign 4"
[image8]: ./examples/Zeichen_34.png "Traffic Sign 5"


---

You're reading it! and here is a link to my [project code](https://github.com/Hyun5/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 1. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![alt text][image0]

It is a bar chart showing how the data are exist in each lable.

![alt text][image1]

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because the color does not matter but it reduce the training time.

Here is an example of a traffic sign image before and after grayscaling.

```
### Converting to grayscale, etc.
X_train_rgb = X_train
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)

X_valid_rgb = X_valid
X_valid_gry = np.sum(X_valid/3, axis=3, keepdims=True)

X_test_rgb = X_test
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)
```

![alt text][image2]

As a last step, I normalized the image data because it can make sure our data looks and reads the same way across all data sets.

Here is an example of an original image and an augmented image:
```
## Normalize the train and test datasets
def preprocess(img):
    return (img - 128.) / 128.

X_train_normalized = np.array([preprocess(img) for img in X_train_gry])
X_valid_normalized = np.array([preprocess(img) for img in X_valid_gry])
X_test_normalized = np.array([preprocess(img) for img in X_test_gry])
```

![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 3x3		| 2x2 stride, outputs 14x14x10  				|
| RELU					|												|
| Convolution 3x3		| 1x1 stride, outputs 8x8x16  					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x16  					|
| Flatten		      	| Size: 256										|
| Fully connected		| Size: 120										|
| RELU					|												|
| Dropout				| 50%											|
| Fully connected		| Size: 100										|
| RELU					|												|
| Fully connected		| Size: 84										|
| RELU					|												|
| Fully connected		| Size: 43										|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used 60 epochs, a batch size of 128 and a learning rate of 0.0009.
Based on some trail, I increased the number of epochs and decreased the learning rate from the initial value.

For my training optimizers I used softmax_cross_entropy_with_logits to get a tensor representing the mean loss value to which I applied tf.reduce_mean to compute the mean of elements across dimensions of the result. Finally I applied minimize to the AdamOptimizer of the previous result.

My final model Validation Accuracy was 0.956

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.956 
* test set accuracy of 0.936



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			       	        		        | Prediction									| 
|:---------------------------------------------:|:---------------------------------------------:| 
| Yield											| Yield											|
| Stop											| Road work										|
| No entry										| No entry										|
| Speed limit (60km/h)					   		| Priority road									|
| Turn left ahead								| Turn left ahead 								|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 
The new image I choosed were very clear and simple whithout any noise and background. I just applied the grayscale and normalized as prepreocessing. I thought it would detect 100% but it did not. Maybe it was because of the overfitting. When I played with serveral hyperparameter values, the test results with new image was better even with the validation result was less good. 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


The top five soft max probabilities were;

```
Zeichen_13.png:
Yield: 100.00%
No vehicles: 0.00%
Ahead only: 0.00%
Priority road: 0.00%
Speed limit (60km/h): 0.00%

Zeichen_14.png:
Road work: 99.96%
Keep right: 0.02%
Stop: 0.01%
Yield: 0.00%
Speed limit (60km/h): 0.00%

Zeichen_17.png:
No entry: 100.00%
No passing: 0.00%
No passing for vehicles over 3.5 metric tons: 0.00%
Priority road: 0.00%
Speed limit (20km/h): 0.00%

Zeichen_3.png:
Priority road: 94.46%
Bicycles crossing: 2.20%
Keep right: 1.72%
Beware of ice/snow: 1.58%
Slippery road: 0.04%

Zeichen_34.png:
Keep right: 53.90%
Turn left ahead: 40.15%
Yield: 5.94%
Stop: 0.00%
No vehicles: 0.00%

```


#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

From the research, the accuray will be better if I use more preprocessing of dataset, such as transaltion, scaling, warp, etc.
I used just 1 dropout and 1 maxpooling. More drop and maxpooling in every layer will increase the accuracy by avoiding the overfitting.

