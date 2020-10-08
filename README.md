# Image Blur Detection 

By Hemanth Pasupuleti

Classifier which classifies whether the given image is blurred or not.
This model created with the help of sklearn and open-cv.
Dataset used: http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip


 
## Intuition
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

I have used sklearn library to implement the SVM classifier.




### Tech
* [OpenCv](https://opencv.org/) - To calculate the laplacian Gradient
* [Pillow](https://python-pillow.org/) - To read the image
* [Sklearn](https://scikit-learn.org/stable/index.html) - To build the Classification Model
* [Matplotlib](https://matplotlib.org/) - To plot the data
* [Pandas](https://pandas.pydata.org/) - For reading csv files and creating dataframes
* [Numpy](https://numpy.org/) - To execute matrix operations
* [Mlxtend](http://rasbt.github.io/mlxtend/) - For plotting the decision boundary


### Installation

```sh
$ pip install opencv-python    #To install OpenCv
$ pip install pillow           #To install Pillow
$ pip install -U scikit-learn  #To install Sklearn
$ pip install -U matplotlib    #To install Matplotlib
$ pip install pandas           #To install Pandas
$ pip install numpy            #To install Numpy
$ pip install mlxtend          #To install Mlxtend

```

