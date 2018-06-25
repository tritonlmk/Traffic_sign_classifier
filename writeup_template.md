# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./examples/visualization.JPG "Visualization"
[image2]: ./examples/grayscale.JPG "Grayscaling"
[image3]: ./examples/random_noise.JPG "Random Noise"
[image4]: ./examples/placeholder.PNG "Traffic Sign 1"
[image5]: ./examples/argumented.PNG "ARGUMENTED_DATA_SET"
[image6]: ./new_images/00006.JPG "NEW_IMAGE_1"
[image7]: ./new_images/00009.JPG "NEW_IMAGE_2"
[image8]: ./new_images/00024.JPG "NEW_IMAGE_3"
[image9]: ./new_images/00051.JPG "NEW_IMAGE_4"
[image10]: ./new_images/00060.JPG "NEW_IMAGE_5"
[image11]: ./new_images/00086.JPG "NEW_IMAGE_6"
[image12]: ./new_images/00093.JPG "NEW_IMAGE_7"
[image13]: ./new_images/00107.JPG "NEW_IMAGE_8"
[image14]: ./examples/result_softmax.JPG "Result_softmax"


## Rubric Points


### Writeup / README


You're reading it! and here is a link to my [project code](https://github.com/tritonlmk/Traffic_sign_classifier/blob/master/TrafficSignClassifier_refined.ipynb)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 
34799
* The size of the validation set is ?
4410
* The size of test set is ?
12630
* The shape of a traffic sign image is ?
32x32
* The number of unique classes/labels in the data set is ?
43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how the image data is preprocessed the image data.

As a first step, I decided to convert the images to grayscale because color is irrelvent with the result of the classifier

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because I want to make the parameters smaller thus results in faster computing  speed

I decided to generate additional data because some classes contain very little pictures compared with others. So we need to argument those classes to make the convolutionnal network perform better 

To add more data to the the data set, I used the functions from lib cv2, to make those pictures different from the one it is generated from and haveing the right classification at the same time. 

The code used here is got form 'Jeremy Shannon's Traffic_Sign_Classifier', for I know very little to image transformation, I can do very little to this part of the code

Here is an example of an original image and an augmented image:

![alt text][image3]

The augmented data set is the following 

![alt text][image5]


#### 2. Describtion of the model used in this program

My final model consisted of the following layers:

| Layer             		|     Description	        	            				| 
|:---------------------:|:---------------------------------------------:| 
| Input         		    | 32x32x3 RGB image   			    		        		| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x12 	  |
| RELU				        	| activation    					                			|
| Max pooling	        	| 2x2 stride,  outputs 14x14x12         				|
| Convolution 3x3	      | 1x1 stride,  outputs 10x10x32 		        		|
| RELU          	      |                                		        		|
| Max Pooling	          | 2x2 stride,  outputs 5x5x32    		        		|
| Flatten       	      | input 5x5x32, outputs 800      	         			|
| Fully connedted	      | input 800,    outputs 240      		        		|
| Fully connected		    | input 240,    outputs 110				          		|
| RELU          	      | activation                             				|
| dropout          	    | dropout_rate = 0.5               			      	|
| Fully connected	    	| input 110,    outputs 43					          	|
| Softmax			        	| etc.      
 


#### 3. Describe the how model is trained

To train the model, I used an learning rate of 0.0007 to run 35 epochs
Then I use an learning rate of 0.0003 to run 95 epochs
The batch size of the two steps mentioned above are all 128

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

I tried the original LeNet but it doesnot work well. then I did some data pre-processing and it still doesn't work well. At last I add the depth of the convolutionnal network and then it works.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 0.960 
* test set accuracy of 0.951

For the changes made to LeNet:
* What was the first architecture that was tried and why was it chosen?
LeNet original structure.
* What were some problems with the initial architecture?
not enough depth, which means the features the convolutionnal network captured is not enough.
* How was the architecture adjusted and why was it adjusted?
add the depth of the LeNet
* Which parameters were tuned? How were they adjusted and why?
Add depth to the LeNet and making the fully connected layer having more inputs
* What are some of the important design choices and why were they chosen?
dropout can help with over-fitting, which will result a very high validation accurancy but relatively low test accurancy

If a well known architecture was chosen:
* What architecture was chosen?
LeNet
* Why did you believe it would be relevant to the traffic sign application?
the original LeNet is used for classifying characters. And for traffic signs
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
the validation accurancy is 0.96, and the test accurancy is 0.95, those two numbers doesn't go too far from each other.
 

### Test a Model on New Images

#### 1. Choose eight German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13] 

The first image might be difficult to classify because ...



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

![alt text][image14]

The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. It's good


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 



