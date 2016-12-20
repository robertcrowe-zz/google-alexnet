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

The data for this project comes from the Google Street View House Numbers 
(SVHN) Dataset:

>_“SVHN is a real-world image dataset for developing machine learning and 
object recognition algorithms with minimal requirement on data preprocessing 
and formatting. It can be seen as similar in flavor to MNIST (e.g., the 
images are of small cropped digits), but incorporates an order of magnitude 
more labeled data (over 600,000 digit images) and comes from a significantly 
harder, unsolved, real world problem (recognizing digits and numbers in 
natural scene images). SVHN is obtained from house numbers in Google Street 
View images.”_