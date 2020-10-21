# Image Blur Detection 

By Hemanth Pasupuleti

Classifier which classifies whether the given image is blurred or not.
This model created with the help of sklearn and open-cv.

#### CERTH Image Blur Dataset
     E. Mavridaki, V. Mezaris, "No-Reference blur assessment in natural images using Fourier transform and spatial pyramids", Proc. IEEE International Conference on Image Processing (ICIP 2014), Paris, France, October 2014.
     
The CERTH Image Blur Dataset consists a total of 2450 undistorted, artificially-blurred and naturally-blurred digital images. 

Dataset: http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip

Extract the files and load them into directory named **CERTH_ImageBlurDataset**.

 

# Intuition
##### Laplacian filter: 

    We can use laplacian filter to get the edges in the image and get the variance and maximum 
    of the laplacian gradient.
    [[0  1 0]
     [1 -4 1]    is the laplacian filter. There are other variants of this filter.
     [0  1 0]]

###### Steps:
 - Import the image
 - Scale it to reduce computation
 - Convert it into GreyScale
 - Get the edges with by doing laplacian of the GreyScale Image
 - Compute variance and maximum of the laplacian gradient
  - If there is high variance then the edges are more spread and we can say that the image is not blurred and for low variance the image will be blurred.
  
Now the real challenge is how do we find a threshold of variance to classify the image as blurred or not.
Here, We can use Machine Learning Algorithms.

##### Support Vector Machine(SVM):

The problem statement we currently have is to classify the given image as Blurred or not with the help of variance and maximum of laplacian gradient per an image.
We can use SVMs to classify our images by taking **Variance** and **Maximum** as features.

To calculate the Variance I have used OpenCV2.

<code>cv2.Laplacian(image,cv2.CV_64F)</code>

I have used sklearn library to implement the SVM classifier.

<code>sklearn.svm.SVC()</code>

On running K-fold cross validation with K = 10 the learning curve is:

![Decision Boundary](https://github.com/Hemanth21k/Image-Blur-Detection/blob/master/Learning_Curve.png)








The final decision boundary achieved is:


![Decision Boundary](https://github.com/Hemanth21k/Image-Blur-Detection/blob/master/Decision_Boundary.png)

This method yielded an accuracy of **87.5%** in the entire evaluation set combined.

### Tech
* [OpenCv](https://opencv.org/) - To calculate the laplacian Gradient
* [Pillow](https://python-pillow.org/) - To read the image
* [Sklearn](https://scikit-learn.org/stable/index.html) - To build the Classification Model
* [Matplotlib](https://matplotlib.org/) - To plot the data
* [Pandas](https://pandas.pydata.org/) - For reading csv files and creating dataframes
* [Numpy](https://numpy.org/) - To execute matrix operations
* [Mlxtend](http://rasbt.github.io/mlxtend/) - For plotting the decision boundary
* [Pickle](https://docs.python.org/3.7/library/pickle.html) - To save the sklearn model


### Installation and Running
```sh
$ pip install opencv-python    #To install OpenCv
$ pip install pillow           #To install Pillow
$ pip install -U scikit-learn  #To install Sklearn
$ pip install -U matplotlib    #To install Matplotlib
$ pip install pandas           #To install Pandas
$ pip install numpy            #To install Numpy
$ pip install mlxtend          #To install Mlxtend

```
After installing the required libraries run the Train.py file to train the image dataset with the command:

<code>python Train.py</code>

The model will be saved to saved_model.pkl after the completion of training.
To run the testing on the evaluation set run the Test.py with the command:

<code>python Test.py</code>      

The evaluation set has 2 different sets of images:

    1) NaturalBlurSet (1000 images)
    2) DigitalBlurSet (480 images) 
    
The outputs labels of these two sets were exported into:

    1) NaturalBlurSet_Predictions1.csv  (82.39% accuracy)
    2) DigitalBlurSet_Predictions1.csv  (98.12% accuracy)
