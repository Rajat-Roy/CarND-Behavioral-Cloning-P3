# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.  

A ipython notebook has also been provided as an execution example.   

Note: Saving model using the ipython notebook is mandatory in windows machine because there is a bug in keras 2.0.9 where it fails to save lambda layers.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

`model.py` contains `Save_Model` function where the NVidia model achitecture has been implemented.

A lambda layer is used to normalize the image. Then a cropping layer removes confusing areas. Then there are 5 convolution layer for image detection. Finally, after a flatten layer, 4 dense layers for regression.
  Following is a table specifying the full network.

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Lambda | lambda x: (x / 255.0) - 0.5 | input_shape=(160,320,3)							|
| Cropping2D | cropping=((70,25), (0,0))									|
| Convolution 5x5  |  24 filters 	| 2x2 stride	| activation='relu'									|
| Convolution 5x5  |  36 filters 	| 2x2 stride	| activation='relu'									|
| Convolution 5x5  |  48 filters 	| 2x2 stride	| activation='relu'									|
| Convolution 3x3  |  64 filters 	| no stride	| activation='relu'									|
| Convolution 3x3 |  64 filters 	| no stride	| activation='relu'									|
| Flatten | 
| Dense	| outputs 100        									|
| Dense		| outputs 50        									|
| Dense		| outputs 10        									|
| Dense		| outputs 1        									|

#### 2. Attempts to reduce overfitting in the model

The model was trained over 2 laps of data.

After several attempts, dropout layers could not improve the practicle performance of driving, so no dropout layers were added.

Instead, only 3-4 epochs were used to train the model to avoid overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I tried to record data using center lane driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use enough data to stay on the middle of the track.

My first step was to use a convolution neural network model similar to the NVIDIA CNN achitecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

Initialy I used 7 epochs training with overfitting.

Then I ran the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.

To improve the driving behavior in these cases, I tried using dropout layers, which produced gradual loss reduction, but the practical performance in the simulator was very poor.

Finally, I tried using 4 epocs to stop further overfitting.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 47-60) consisted of a convolution neural network with the following layers and layer sizes. 

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Lambda | lambda x: (x / 255.0) - 0.5 | input_shape=(160,320,3)							|
| Cropping2D | cropping=((70,25), (0,0))									|
| Convolution 5x5  |  24 filters 	| 2x2 stride	| activation='relu'									|
| Convolution 5x5  |  36 filters 	| 2x2 stride	| activation='relu'									|
| Convolution 5x5  |  48 filters 	| 2x2 stride	| activation='relu'									|
| Convolution 3x3  |  64 filters 	| no stride	| activation='relu'									|
| Convolution 3x3 |  64 filters 	| no stride	| activation='relu'									|
| Flatten | 
| Dense	| outputs 100        									|
| Dense		| outputs 50        									|
| Dense		| outputs 10        									|
| Dense		| outputs 1        									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on `track one` using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

After the collection process, I had 14232 number of data points. I then preprocessed this data by using a Lambda layer in keras to normalize the images. Then I cropped the data using Cropping2D Layer, which crops the upper part of the image where there is no track and contains confusing data. It also crops the bootom part where only the car-front is visible.

I finally randomly shuffled the data set and put 2% (3558) of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4. I used an adam optimizer so that manually training the learning rate wasn't necessary.
