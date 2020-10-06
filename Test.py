import numpy as np
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import joblib



def accuracy_metric(actual, predicted):   #To calculate the accuracy of predicted outputs
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


def getPredictions(clf1,filename,names):
	Pred=[]
	test_path = "CERTH_ImageBlurDataset/EvaluationSet"
	labels = ["Not Blurred","Blurred"]
	i=0

	for name in names:
		i+=1
		path = test_path+"/"+filename+"/"+name
		x = Image.open(path).convert('L')
		n = np.array(x)
		resized = cv2.resize(n, dim, interpolation = cv2.INTER_AREA)
		lap = cv2.Laplacian(resized,cv2.CV_64F)
		variance = lap.var()
		maximum = np.max(lap)
		p2 = clf1.predict([[variance,maximum]])
		Pred.append(p2)
		print(path,"-",i,' - var:',variance,' max:',maximum,' Prediction:',labels[int(p2)])
	return	Pred


clf1 = joblib.load("saved_model.pkl")  #load the model

dim = (400,600)   #set dimensions  

test1 = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx')

names = test1["MyDigital Blur"].tolist()
Y_Test = test1["Unnamed: 1"].values.tolist()


D_Pred1 = getPredictions(clf1,"DigitalBlurSet",names)




test2 = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx')

names2 = test2["Image Name"].tolist()
Y_Test2 = test2["Blur Label"].values.tolist()

for i in range(len(names2)):
	names2[i] = names2[i]+".jpg"


D_Pred2= getPredictions(clf1,"NaturalBlurSet",names2)



final_Test1= []
for i in range(len(D_Pred1)):
    if D_Pred1[i] == 0:
        final_Test1.append(-1)
    else:
        final_Test1.append(1)


print("\nThe accuracy on DigitalBlurSet: ",accuracy_metric(Y_Test,final_Test1))

final_Test2= []
for i in range(len(D_Pred2)):
    if D_Pred2[i] == 0:
        final_Test2.append(-1)
    else:
        final_Test2.append(1)


print("\nThe accuracy on NaturalBlurSet: ",accuracy_metric(Y_Test2,final_Test2))

test1["Prediction"] = final_Test1
test1.to_csv('DigitalBlurSet_Predictions1.csv', index = False) 

test2["Prediction"] = final_Test2
test2.to_csv('NaturalBlurSet_Predictions1.csv', index = False) 
