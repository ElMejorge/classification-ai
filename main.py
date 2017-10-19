import pandas as pd
import scipy
import random
import csv
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

# 1-. Handle data
dataset = loadCsv()
trainDS, testDS = splitDataset(dataset)
# 2-. Summarize data
