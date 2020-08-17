# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/model_architecture/lenet.png "LeNet-5 Model Visualization"
[image2]: ./writeup/center_lane/center_2017_08_03_11_27_02_625.jpg "Centel Raw"
[image3]: ./writeup/recovery/center_2017_08_04_18_06_51_725.jpg "Recovery 1"
[image4]: ./writeup/recovery/center_2017_08_04_18_06_54_359.jpg "Recovery 2"
[image5]: ./writeup/recovery/center_2017_08_04_18_06_57_272.jpg "Recovery 3"
[image6]: ./writeup/flip/original.jpg "Normal Image"
[image7]: ./writeup/flip/flipped.jpg "Flipped Image"

### Files

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The first working model is simply a LeNet-5 model which consists of 2 convolution neural network with 5x5 filter sizes and depths between 6 and 16 (model.py lines 82 and 84).

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 80).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 88 and 90).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 124). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 97).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. To make full use of the data, both the center images and the side images are used and the left images and right images need to be shifted to center view.

We will talk detail about the process to obtain the training data in the next section.

### Pipeline

#### 1. Initial Model Design

The overall strategy is to generalize the association of the images captured by the cameras mounted in the self-driving car and the steering angles.

**LeNet-5** model was chosen as a starting point due to its simplicity and successful application on other image classification problems.

### 2. Data Collecting

To capture a good driving behavior, our experience was to record a few more laps than one or two and in both clockwise and counterclockwise directions when driving in the lane center. An example image of center lane driving can be shown as below,

![alt text][image2]

Some boundary cases were needed to be included such as recovering from the left side and right side of the road back to center so that the vehicle would learn to recover from sticking to the side. These images show what a recovery looks like starting from right boundary of the road,

![alt text][image3]

![alt text][image4]

![alt text][image5]

The other case was that, we recorded a lap focusing on driving smoothly around curves where the car were shown to be close to go off the track.

The process was repeated on Track two in order to get more data points.

### 3. Data Preprocessing

The images collected in the previous step then should be preprocessed by Gaussian Blurring. The images were converted from BGR to RGB format here, and the reason was that OpenCV read images in BGR format, while the following procedures in `drive.py` were expecting the images in RGB format.

### 4. Data Augmentation

To augment the data, multiple techniques could be used while here we flipped images. This was in consideration that the labels or angles could be evenly distributed and avoid bias to one side. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

### 5. Testing in Simulator

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.

To improve the driving behavior in these cases, we can add in more data focusing on moving around the curves and increase the number of the epochs.

In the end, the vehicle is able to drive autonomously around the track without leaving the road.

#### 5. Final Model Architecture

The final model architecture consisted of two convolution layers and three fully connected layers with the following layers and layer sizes.

Here is a visualization of the architecture.

![alt text][image1]

#### 6. Training Process

In order to gauge how well the model was working, the data were splitted into a training and validation set with the proportion of 4:1 and were shuffled.

To combat the potential overfitting, two dropout layers were inserted with keep ratio 0.5.

The training data were used for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 12 as evidenced by practice. I used an `adam` optimizer so that manually training the learning rate wasn't necessary.

## Result Analysis

### Model Assessment

|       Model        |                Setting                  |   1st Track   |         2nd Track       |
|--------------------|-----------------------------------------|---------------|-------------------------|
| ./model/model01.h5 | center camera only, LetNet-5, 5 epochs  |    Perfect    | Go off the track soon.  |
| ./model/model02.h5 | center camera only, LetNet-5, 20 epochs |    Perfect    | Good. Go off the track one time in a lap. |
| ./model/model03.h5 | three cameras, LetNet-5, 12 epochs      |    Perfect    | Perfect for several laps. Possibly go off the track when it runs in a high speed |
