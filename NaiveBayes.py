# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:22:42 2019

@author: Prady
"""

import numpy as np
import pandas as pd
import math
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv("C:\\Users\\curaj\\Desktop\\iris.data",header=None)
#iris.Species.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [1, 2, 3], inplace=True)
df.iloc[:,4]=df.iloc[:,4].map({'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3})
train, test = train_test_split(df, test_size=0.4)
train=np.array(train,dtype=float)
test=np.array(test,dtype=float)
#X=train.iloc[:,0:5]
#X_train=np.array(X,dtype=float)
#Y=train.iloc[:,4]
#Y_train=np.array(Y,dtype=float)

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
    
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def prob(x,m,s):
    #l0=x.shape[0]
    #pi=math.pi
    #mv=[m]*l0
    #mv=pd.DataFrame(mv)
    return(1/(np.sqrt(2*math.pi)*s))*(np.exp(-((1/2)*(((x-m)/s)**2))))

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= prob(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

summaries = summarizeByClass(train)
	# test model
predictions = getPredictions(summaries, test)
accuracy = getAccuracy(test, predictions)
print(predictions)
print(accuracy)

#plt.plot()