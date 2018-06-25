# Traffic_sign_classifier
Traffic_sign_classifier using LeNet structure

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

*The size of the training set is 34799

*Ths size of the validation set is 4410

*The size of teh test set is 12630

*The shape of all the traffic signs are transformed int to 32*32

*The numbe of total classes are 43


PreProcess of the Data Set:

grayscale of the Data set and than change the shape into (*,32,32,1)

normalize the data set into [-1，1]

Using existing data to produce fake data to argument the original data set, for there are too few data in some classes
"original data used in this part from jeremy-shannon's traffic sign calssifier.ipyb"

Training Model:LeNet



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


Batch size = 128
learning rate 0.0007 epochs:35
learning rate 0.0003 epochs:95

final valication accurancy: 0.960
final test rate: 0.9509

showing the new images and analye prformance:

accurancy for the new images downloaded from Graman Traffic Signs is 0.875

The problem of the initial LeNet is that the convolution layser output is not deep enough, which means that it does not extract enough features from the original picture. This may because the original LeNet is deigned to focus on letter recognition originally, which has only 10 classes in total.

For the traffic sign classifier, it is also a classifier, only with more features and more output classes, so I simply deepen the output deepth of the convolution layers to make it more accurate, also I add a dropout layer which aims to prevent overfitting.


