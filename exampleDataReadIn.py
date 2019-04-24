import numpy as np
from random import random
from math import exp
import copy
import csv


def loadCSV(fileName):
    data = np.loadtxt(fileName, dtype=float, delimiter=",")
    print(fileName, " loaded in successfuly")
    return data


TestDigitX = loadCSV("TestDigitX2.csv.gz")
print(len(TestDigitX[0]))
print(len(TestDigitX))
count = 1
for i in range(784):
    if TestDigitX[2][i] > 0.1:
        print('1', end='')
    else:
        print('0', end='')
    if count == 28:
        count = 0
        print()
    count += 1
