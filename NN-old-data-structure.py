from random import random
from math import exp
import copy
import csv
import numpy as np

def loadCSV(fileName):
    data = np.loadtxt(fileName, dtype=float, delimiter=",")
    print(fileName, " loaded in successfuly")
    return data

def sigmoidPrime(output):
    return output * (1.0 - output)

def sigmoid(s):
    return 1.0 / (1.0 + exp(-s))

def initialize_network(NInput, NHidden, NOutput):
    network = list()
    hiddenNueron = [{'weights':[random() for i in range(NInput)]} for i in range(NHidden)]
    outNueron = [{'weights':[random() for i in range(NHidden)]} for i in range(NOutput)]
    network.append(hiddenNueron)
    network.append(outNueron)
    return network

def printNetwork(network):
    print("Hidden Layer")
    for i in range(len(network[0])):
        for k in range(len(network[0][i]['weights'])):
            print("Input node: ", i, " to Hidden node: ", k, "weight = ", network[0][i]['weights'][k])
    print()
    print("Output Layer")
    for i in range(len(network[1])):
        for k in range(len(network[1][i]['weights'])):
            print("Input node: ", i, " to Hidden node: ", k, "weight = ", network[0][i]['weights'][k])

def weightedSum(weights, inputs):
    sum = weights[-1]
    print(sum)
    for i in range(len(weights)-1):
        sum += weights[i] * inputs[i]
    return sum

def forwardPass(network, inputs):
    for layer in network:
        newInputs = []
        for neuron in layer:
            activation = round(weightedSum(neuron['weights'], inputs),8)
            neuron['output'] = sigmoid(activation)
            newInputs.append(neuron['output'])
        inputs = newInputs

    return newInputs

#TestDigitX = loadCSV("TestDigitX2.csv.gz")
#network = initialize_network(2,2,2)
network=[[  {'weights': [0.1,0.1,0.1]}, #[w1, w3, w9]
		    {'weights': [0.2,0.1,0.1]}], #[w2, w4, w10]
        [   {'weights': [0.1,0.1,0.1]},  #[w5, w7, w11]
            {'weights': [0.1,0.2,0.1]}]] #[w6, w8, w12]

X1 = [[0.1, 0.1, 1],[0.1,0.2,1]]
Y1 = [[1,0],[0,1]]
m = 2
learningRate = 0.1
epochs = 1
printNetwork(network)
print(forwardPass(network,X1[1] ))
print(network[0][0])
