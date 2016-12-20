# google-alexnet
An implementation of AlexNet in Tensorflow to classify Google Street View 
House Numbers

Robert Crowe, November 2016

## Project Overview
Learning and recognizing text in images is an application of deep 
learning with a wide range of uses.  This project addresses a subset 
of text recognition by recognizing single digits from street addresses 
in outdoor settings using the Google Street View House Numbers dataset.  

This is important for accurate mapping, for example when used with the 
Google Street View capture system, by helping to verify and improve 
existing maps by more accurately connecting addresses with specific 
locations.  The importance of this task for accurate mapping has led a 
[Google team] (https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf)
to also attempt the same task.  It is also a tractable example 
of a real-world computer vision problem with practical applications.

The data for this project comes from the [Google Street View House Numbers 
(SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/):

>_“SVHN is a real-world image dataset for developing machine learning and 
object recognition algorithms with minimal requirement on data preprocessing 
and formatting. It can be seen as similar in flavor to MNIST (e.g., the 
images are of small cropped digits), but incorporates an order of magnitude 
more labeled data (over 600,000 digit images) and comes from a significantly 
harder, unsolved, real world problem (recognizing digits and numbers in 
natural scene images). SVHN is obtained from house numbers in Google Street 
View images.”_

## Implementation

Tensorflow v10 was used on an AWS p2.8xlarge instance, with Cuda 7.5 and cuDNN v.5.  
Training with 100 epochs required 6,345 seconds (~105 minutes).  Working in Python, 
I used the TFlearn library on top of Tensorflow to simplify the coding effort, and 
found it to be well-suited to this project, along with Numpy and Scikit-Learn.

The [original AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 
was designed for input images of 227x227x3, but the SVHN dataset 
is 32x32x3, so a significant task was to reshape AlexNet for a smaller input size 
while trying to preserve the original AlexNet design, with similar changes in the 
visual fields between layers.  This involved changing the filter sizes and strides, 
for both 2-D convolutional layers and Max Pool layers.

My intuitive view of the difference in complexity of the visual features in the 
ImageNet dataset that the original AlexNet was designed for, versus the SVHN dataset 
that I am using, is that ImageNet is significantly more complex.  ImageNet has 
1,000 object categories, whereas SVHN has only 10.  This suggests that the number 
of filters required in 2-D convolutional layers to recognize important features 
should be significantly lower for SVHN, so I also made adjustments to the numbers 
of filters (aka “kernels”, “kernel maps”, or “feature maps”).  This resulted in a 
significant decrease in resource requirements, with a small if any decrease in 
accuracy.

### References:
Krizhevsky, Sutskever, Hinton _"ImageNet Classification with Deep Convolutional Neural Networks"_, 2012

https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

### Prerequisites:
	Tensorflow: https://www.tensorflow.org/
	TFlearn: http://tflearn.org/
	Scikit-Learn: http://scikit-learn.org/
	NumPy: http://www.numpy.org/
	Pandas: http://pandas.pydata.org/

### Dataset:
	http://ufldl.stanford.edu/housenumbers/

### Code:
	cnn.py - main code
	pics.py - for generating misclassified example images
	confusion.py - for generating the confusion matrices