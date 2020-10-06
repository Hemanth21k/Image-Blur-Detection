import numpy as np
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import joblib


# Training

def accuracy_metric(actual, predicted):   #To calculate the accuracy of predicted outputs
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


def getFeatures(data,filename,label):
    train_path = "CERTH_ImageBlurDataset/TrainingSet"
    i = 0
    print("Opening" +filename+" file: \n")
    for file in glob.glob("{}/*".format(train_path+"/"+filename)):
        i+=1
        x = Image.open(file).convert('L')
        n = np.array(x)
        resized = cv2.resize(n, dim, interpolation = cv2.INTER_AREA)
        lap = cv2.Laplacian(resized,cv2.CV_64F)
        variance = lap.var()
        maximum = np.max(lap)
        data.append([variance,maximum,label])
        print(file,"-",i,' - var:',variance,' max:',maximum)
    return 

data = []


dim = (400,600)

print("Images are being scaled to 400X600 dimensions...\n")

getFeatures(data,"Undistorted",-1)
getFeatures(data,"Naturally-Blurred",1)
getFeatures(data,"Artificially-Blurred",1)



df = pd.DataFrame(data)
df.columns = ["variance","maximum","label"]


df.to_csv('Train_max_var.csv', index = False) 


# Testing

skdata = []

df = pd.read_csv('Train_max_var.csv')

X = df[['variance','maximum']].to_numpy()

Y = df['label'].to_numpy()

Y10 = []
for i in range(len(Y)):
    if Y[i] == -1:
        Y10.append(0)
    else:
        Y10.append(1)


X_S,Y_S = shuffle(X,Y10,random_state=0)

svc1 = SVC(gamma='auto')
clf1 = make_pipeline(StandardScaler(), svc1)

clf1.fit(X_S,Y_S)

print("\nScore on Training set: ",clf1.score(X_S,Y_S))

joblib.dump(clf1,"saved_model.pkl")  #save the trained model'
print("\n The model is saved in saved_model.pk1 file...")