# **Traffic Sign Recognition** 



---

## **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/sample.jpg "sample"
[image3]: ./examples/grayscale.jpg "Grayscaling"
[image4]: ./examples/noise.jpg "noise"
[image5]: ./images/0.ppm "Speed limit (20km/h)"
[image6]: ./images/12.ppm "Priority road"
[image7]: ./images/15.ppm "No vehicles"
[image8]: ./images/16.ppm "Vehicles over 3.5 metric tons prohibited"
[image9]: ./images/27.ppm "Pedestrians"


### Summary

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32x32x3)
* The number of unique classes/labels in the data set is 43

#### Visualization

Here is an exploratory visualization of the data set.

![alt text][image1]

### Preprocess

As a first step, I decided to convert the images to grayscale to decrease the computational cost and to normalize to obtain a better optimization problem. 

Here is an example of a traffic sign image before the preprocessing

![alt text][image2]

and after

![alt text][image3]

### Data Augmentation

I decided to generate additional data to increase the train set size and to balance the classes: in utils.py I used tensorflow to generate noisy and transformed(rotate, inverted, ecc...) version of the original images in the less frequent classes: because this task is time consuming I parallelized the process.

As an example of random noise

![alt text][image4]

I accumulated other 2k images to train the model.
Using this script it is possible to obtain a perfectly balance dataset but for the purpose of this assignment it seems no useful. However for the final prediction I didn't used this augmented data because I obtained the best accuracy using only the original data.


### Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale and normalized image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 
| RELU					|											
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x32 
| RELU					|											
| Max pooling	      	| 2x2 stride,  outputs 5x5x32			
| Flatten 				| input 5x5x32 = 800
| Fully connected		| hidden units 400
| RELU					| 
| Dropout				| 0.8
| Fully connected		| hidden units 200
| RELU					| 
| Dropout				| 0.8
| Fully connected		| output n_classes
| Softmax				|								|



### Setting

To train the model I used the Adam optimizer and the following hyper-parameters:

| Hyper-parameters         		|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| EPOCHS 				| 20
| BATCH_SIZE        		| 64  
| learning_rate 			| 0.001
| decay_rate	     		| 0.99
| prob_dropout			| 0.8										

### Results

My final model results were:
* training set accuracy of 0.99
* validation set accuracy of 0.94 
* test set accuracy of 0.92

I tried different architecture starting with the LeNet and modifying the hyper-parameters (more hidden units, different learning rate, ecc..).
To increase the accuracy on the validation set (overfitting problem) I decided to use a dropout regularization on the fully-connected layers.

The final result could be improved adding EPOCHS or increasing the network.
 

### Test a Model on New Images



Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)      		| Speed limit (20km/h)
| Priority road     			| Priority road
| No vehicles					| No vehicles
| no Vehicles over 3.5	      		| no Vehicles over 3.5
| Pedestrians			| Pedestrians     							


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. Obviously this sample is to small to infer conclusion, but validate our result on the test set.

### 5 top-max

The code for making predictions on my final model is located in the 4th cell (starting to count from the last cell) of the Ipython notebook.

For example for the pedestrian image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99202788e-01         			| Pedestrians  					
| 7.83480296e-04     				| Road narrows on the right
| 8.86880025e-06					| Right-of-way at the next intersection	
| 2.18602531e-06	      			| Double curve
| 9.28980967e-07				    | Children crossing