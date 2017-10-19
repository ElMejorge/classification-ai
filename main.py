import pandas as pd
import scipy
import random
import csv
import math
from pandas import DataFrame, Series

#  Obtain dataset as array
def loadCsv():
    filename = "diabetes.data.csv"
    lines = csv.reader(open(filename,"rt"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

# Divide dataset into train dataset and test data set.
# 2/3 an 1/3 is a common ratio for training and testing datasets
def splitDataset(dataset):
    ratio = 0.67
    trainDSLength = int(ratio*len(dataset))
    trainDS = []
    testDS = list(dataset)
    while len(trainDS) < trainDSLength:
        index = random.randrange(len(testDS))
        trainDS.append(testDS.pop(index))
    return [trainDS, testDS]

def separateByClass(dataset):
    classes = {}
    for i in range(len(dataset)):
        element = dataset[i]
        if (element[-1] not in classes): #Get last attribute in element (which is the class variable (If subject has diabetes or not)
            classes[element[-1]] = []
        classes[element[-1]].append(element)
    return classes

def getMean(dataset):
    return float(sum(dataset)/len(dataset))

def getStdDev(dataset):
    return None

# 1-. Handle data
dataset = loadCsv()
trainDS, testDS = splitDataset(dataset)
# 2-. Summarize data
classes = separateByClass(dataset)


