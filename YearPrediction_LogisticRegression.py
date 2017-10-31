#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:49:20 2017

@author: guptasa
"""
def executeAlgo( datasetDirectory , codebaseDir ):
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd
	import os
	
	#os.chdir("/users/pgrad/guptasa/Downloads")
	
	# Importing the dataset
	newDir = datasetDirectory+'YearPredictionMSD.txt'
	print("Dataset Being Used:",newDir)
	dataset = pd.read_csv(newDir, header=-1)
	
	dataset['old_or_recent'] = np.where(dataset[0]>=2000, '1', '0')
	
	
	
	D1 = dataset.iloc[0:100,:].values
	D1=pd.DataFrame(D1)
	
	D2 = dataset.iloc[0:500,:].values
	D2=pd.DataFrame(D2)
	
	D3 = dataset.iloc[0:1000,:].values
	D3=pd.DataFrame(D3)
	
	D4 = dataset.iloc[0:5000,:].values
	D4=pd.DataFrame(D4)
	
	D5 = dataset.iloc[0:10000,:].values
	D5=pd.DataFrame(D5)
	
	D6 = dataset.iloc[0:50000,:].values
	D6=pd.DataFrame(D6)
	
	D7 = dataset.iloc[0:100000,:].values
	D7=pd.DataFrame(D7)
	
	D8 = dataset.iloc[0:500000,:].values
	D8=pd.DataFrame(D8)
	
	D9 = dataset.iloc[0:1000000,:].values
	D9=pd.DataFrame(D9)
	
	D10 = dataset.iloc[0:5000000,:].values
	D10 =pd.DataFrame(D10)
	
	print(" Start : For D1===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D1.iloc[:, 1:91].values
	y1 = D1.iloc[:, 91].values
	
	#Diving the dataset through 10 KFolds
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	print(kf.get_n_splits(X1))
	
	from sklearn.metrics import mean_squared_error,r2_score,precision_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
	from math import sqrt
	
	#Initializing the metric values
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = LogisticRegression(random_state = 0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
	
	
	
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)   #Precision Score
		mean = mean + precision_score(y1_test, y1_pred,average='macro')               #Precision Score
	
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	
	
	
	print("Mean Precision Score",mean)
	print("Mean Accuracy Score",meanacc)
	print("End: For D1===================================================\n\n")
	
	
	
	print(" Start : For D2===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D2.iloc[:, 1:91].values
	y1 = D2.iloc[:, 91].values
	
	#Diving the dataset through 10 KFolds
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	print(kf.get_n_splits(X1))
	
	from sklearn.metrics import mean_squared_error,r2_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
	from math import sqrt
	
	#Initializing the metric values
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = LogisticRegression(random_state = 0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
	
	
		#meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)   #Accuracy Score
		mean = mean + precision_score(y1_test, y1_pred,average='macro')               #Precision Score
		#variance = variance + regressor.score(X1_test, y1_test)
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	
	
	
	print("Mean Precision Score",mean)
	
	print("Mean Accuracy Score",meanacc)
	
	print("End: For D2===================================================\n\n")
	
	print(" Start : For D3==================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D3.iloc[:, 1:91].values
	y1 = D3.iloc[:, 91].values
	
	#Diving the dataset through 10 KFolds
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	print(kf.get_n_splits(X1))
	
	from sklearn.metrics import mean_squared_error,r2_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
	from math import sqrt
	
	#Initializing the metric values
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = LogisticRegression(random_state = 0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
	
	
		#meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)   #Accuracy Score
		mean = mean + precision_score(y1_test, y1_pred,average='macro')               #Precision Score
		#variance = variance + regressor.score(X1_test, y1_test)
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	
	
	
	print("Mean Precision Score",mean)
	
	print("Mean Accuracy Score",meanacc)
	
	print("End: For D3===================================================\n\n")
	
	print(" Start : For D4===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D4.iloc[:, 1:91].values
	y1 = D4.iloc[:, 91].values
	
	#Diving the dataset through 10 KFolds
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	print(kf.get_n_splits(X1))
	
	from sklearn.metrics import mean_squared_error,r2_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
	from math import sqrt
	
	#Initializing the metric values
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = LogisticRegression(random_state = 0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
	
	
		#meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)   #Accuracy Score
		mean = mean + precision_score(y1_test, y1_pred,average='macro')               #Precision Score
		#variance = variance + regressor.score(X1_test, y1_test)
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	
	
	
	print("Mean Precision Score",mean)
	
	print("Mean Accuracy Score",meanacc)
	
	print("End: For D4===================================================\n\n")
	
	print(" Start : For D5===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D5.iloc[:, 1:91].values
	y1 = D5.iloc[:, 91].values
	
	#Diving the dataset through 10 KFolds
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	print(kf.get_n_splits(X1))
	
	from sklearn.metrics import mean_squared_error,r2_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
	from math import sqrt
	
	#Initializing the metric values
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = LogisticRegression(random_state = 0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
	
	
		#meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)   #Accuracy Score
		mean = mean + precision_score(y1_test, y1_pred,average='macro')               #Precision Score
		#variance = variance + regressor.score(X1_test, y1_test)
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	
	
	
	print("Mean Precision Score",mean)
	
	print("Mean Accuracy Score",meanacc)
	
	print("End: For D5===================================================\n\n")
	
	print(" Start : For D6===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D6.iloc[:, 1:91].values
	y1 = D6.iloc[:, 91].values
	
	#Diving the dataset through 10 KFolds
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	print(kf.get_n_splits(X1))
	
	from sklearn.metrics import mean_squared_error,r2_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
	from math import sqrt
	
	
	#Initializing the metric values
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = LogisticRegression(random_state = 0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
	
	
		#meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)   #Accuracy Score
		mean = mean + precision_score(y1_test, y1_pred,average='macro')               #Precision Score
		#variance = variance + regressor.score(X1_test, y1_test)
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	
	
	
	print("Mean Precision Score",mean)
	
	print("Mean Accuracy Score",meanacc)
	
	print("End: For D6===================================================\n\n")
	
	
	print(" Start : For D7===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D7.iloc[:, 1:91].values
	y1 = D7.iloc[:, 91].values
	
	#Diving the dataset through 10 KFolds
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	print(kf.get_n_splits(X1))
	
	from sklearn.metrics import mean_squared_error,r2_score,precision_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
	from math import sqrt
	
	#Initializing the metric values
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = LogisticRegression(random_state = 0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
	
	
		#meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)   #Accuracy Score
		mean = mean + precision_score(y1_test, y1_pred,average='macro')               #Precision Score
		#variance = variance + regressor.score(X1_test, y1_test)
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	
	
	
	print("Mean Precision Score",mean)
	
	print("Mean Accuracy Score",meanacc)
	
	print("End: For D7===================================================\n\n")
	
	
	
	print(" Start : For D8===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D8.iloc[:, 1:91].values
	y1 = D8.iloc[:, 91].values
	
	#Diving the dataset through 10 KFolds
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	print(kf.get_n_splits(X1))
	
	from sklearn.metrics import mean_squared_error,r2_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
	from math import sqrt
	
	
	#Initializing the metric values
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = LogisticRegression(random_state = 0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
	
	
		#meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)   #Accuracy Score
		mean = mean + precision_score(y1_test, y1_pred,average='macro')               #Precision Score
		#variance = variance + regressor.score(X1_test, y1_test)
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	
	
	
	print("Mean Precision Score",mean)
	
	print("Mean Accuracy Score",meanacc)
	
	print("End: For D8===================================================\n\n")
	
	print(" Start : For D9===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D9.iloc[:, 1:91].values
	y1 = D9.iloc[:, 91].values
	
	#Diving the dataset through 10 KFolds
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	print(kf.get_n_splits(X1))
	
	from sklearn.metrics import mean_squared_error,r2_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
	from math import sqrt
	
	#Initializing the metric values
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = LogisticRegression(random_state = 0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
	
	
		#meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)   #Accuracy Score
		mean = mean + precision_score(y1_test, y1_pred,average='macro')               #Precision Score
		#variance = variance + regressor.score(X1_test, y1_test)
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	
	
	
	print("Mean Precision Score",mean)
	
	print("Mean Accuracy Score",meanacc)
	
	print("End: For D9===================================================\n\n")
	
	
	
	print(" Start : For D10===================================================\n\n")
	
	#creating a matrix of features and target feature
	X1 = D10.iloc[:, 1:].values
	y1 = D10.iloc[:, 0].values
	
	#Diving the dataset through 10 KFolds
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=10,shuffle=True,random_state=0)
	print(kf.get_n_splits(X1))
	
	from sklearn.metrics import mean_squared_error,r2_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
	from math import sqrt
	
	#Initializing the metric values
	mean =0
	meanacc = 0
	count =0
	
	for train_index, test_index in kf.split(X1):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X1_train, X1_test = X1[train_index], X1[test_index]
		y1_train, y1_test = y1[train_index], y1[test_index]
		#print("X1 Train" , X1_train)
		#print("X1_test" , X1_test)
		#print("y1_train" , y1_train)
		#print("y1_test" , y1_test)
		regressor = LogisticRegression(random_state = 0)
		regressor.fit(X1_train, y1_train)
		y1_pred = regressor.predict(X1_test)
		y1_test = y1_test.astype(int)
		y1_pred = y1_pred.astype(int)
	
	
		#meanacc = meanacc + accuracy_score(y1_test, y1_pred)      #Accuracy
		meanacc = meanacc + accuracy_score(y1_test, y1_pred)   #Accuracy Score
		mean = mean + precision_score(y1_test, y1_pred,average='macro')               #Precision Score
		#variance = variance + regressor.score(X1_test, y1_test)
		count = count +1
		#print("Y1 TEST", y1_test)
		#print("Y1 PRED", y1_pred)
		#break
	mean =mean/count;
	meanacc = meanacc/count;
	
	
	
	print("Mean Precision Score",mean)
	
	print("Mean Accuracy Score",meanacc)
	
	print("End: For D10===================================================\n\n")
	return 1