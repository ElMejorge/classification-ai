import pandas as pd
import scipy
import csv
from pandas import DataFrame, Series

# Handle data
def loadCsv():
    lines = csv.reader(open("diabetes.data.csv", "rt"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    print('Loaded data file {0} with {1} rows'.format("diabetes", len(dataset)))
    return dataset

loadCsv()
