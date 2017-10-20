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
    average = getMean(dataset)
    variance = sum([pow(x-average,2) for x in dataset])/float(len(dataset)-1)
    return math.sqrt(variance)

def getSummary(dataset):
    attributes = zip(*dataset)
    summaries = []
    for attribute in attributes:
        summaries.append([getMean(attribute),getStdDev(attribute)])
    del summaries[-1]   # Delete class variable
    return summaries

def summarizeAttributes(classes):
    summaries = {}
    for classValue, elements in classes.items():
        summaries[classValue] = getSummary(elements)
    return summaries

# Formula to calculate if attribute value from element is more likely to be from a class variable
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

# 1-. Handle data
dataset = loadCsv()
trainDS, testDS = splitDataset(dataset)
# 2-. Summarize data
classes = separateByClass(dataset)
summaries = summarizeAttributes(classes)
# 3-. Make predictions
correct = 0
failure = 0
for testElement in testDS:
    result = predict(summaries, testElement)
    if(result == testElement[-1]):
        correct += 1
    else:
        failure += 1
    # print('Prediction: {0}\t Solution {1}'.format(result, testElement[-1]))
print('Correct predictions: {0}\t Failed: {1}'.format(correct, failure))
print('Accuracy: {0}'.format(correct/len(testDS)))

