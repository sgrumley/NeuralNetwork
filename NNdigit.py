from random import random
from math import exp
import copy
import csv
import numpy as np

def loadCSV(fileName):
    data = np.loadtxt(fileName, dtype=float, delimiter=",")
    print(fileName, " loaded in successfuly")
    return data

def load_data_wrapper():
    training_inputs = loadCSV('TrainDigitX.csv.gz')
    results = loadCSV("TrainDigitY.csv.gz")
    training_results = [vectorized_result(y) for y in results] # vectorised results
    training_data = list(zip(training_inputs, training_results))
    test_inputs = loadCSV("TestDigitX.csv.gz")
    testResults =  loadCSV("TestDigitY.csv.gz")
    test_data = list(zip(test_inputs, testResults))
    return (training_data, test_data)

class Neural_Network(object):
    def __init__(self, Ninput, Nhidden, Noutput):
        #parameters
        self.inputSize = Ninput
        self.outputSize = Noutput
        self.hiddenSize = Nhidden

        #weights
        #W1 = input -> hidden layer weights
        #W2 =  hidden layer -> output weights
        self.W1 = np.random.rand(self.inputSize, self.hiddenSize)
        # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.rand(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
        self.bias1 = np.random.rand(self.hiddenSize)
        self.bias2 = np.random.rand(self.outputSize)


    def findError(self, y, X):
        o = NN.forward(X)
        meanSquare1 = 0.5*(y[0] -  o[0])**2
        meanSquare2 =0.5*(y[1] -  o[1])**2
        return meanSquare1 + meanSquare2

    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) + self.bias1# dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2)+ self.bias2 # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3) # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)


    def backPropLayer2(self, X, y,o, currentM):
        """ Second Layer """
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error
        #tmp = self.z2 * self.o_delta * -1
        #self.errorToNode = np.append(tmp, self.z2[::-1] * self.o_delta * -1)
        """ First Layer """
        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error
        tmp = X * self.z2_delta * -1
        self.errorToNode2 = np.append(tmp, X[::-1] * self.z2_delta * -1)

        if currentM == 0:
            self.summedBias2 = self.o_delta *-1
            self.summedBias1 = self.z2_delta *-1
            self.summedLayer2 = self.errorToNode
            self.summedLayer1 = self.errorToNode2
        else:
            self.summedLayer2 += self.errorToNode
            self.summedLayer1 += self.errorToNode2
            self.summedBias2 += self.o_delta *-1
            self.summedBias1 +=  self.z2_delta *-1



    def UpdateWeights(self, n, m):
        """ update weights  """
        #update first layer of weights
        count = 0
        for i in range(len(self.W1)):
            for j in range(len(self.W1)):
                newWeight = (self.summedLayer1[count] * n) / m
                self.W1[i][j] -= newWeight
                count += 1
        #update second layer of weights
        count = 0
        for i in range(len(self.W2)):
            for j in range(len(self.W2)):
                newWeight = (self.summedLayer2[count] * n) / m
                self.W2[i][j] -= newWeight
                count += 1
        #Update Bias
        for i in range(len(self.bias1)):
            newWeight = (self.summedBias1[i] * n) / m
            self.bias1[i] -= newWeight

        for i in range(len(self.bias2)):
            newWeight = (self.summedBias2[i] * n) / m
            self.bias2[i] -= newWeight

    def miniBatch(self,n,m,mX,my):
        for i in range(m):
            """ forward propergate """
            resultForward = NN.forward(mX[i])
            """ Back propergate through last layer """
            NN.backPropLayer2(mX[i],my[i],resultForward, i)




mX = np.array(([0.1, 0.1],[0.1,0.2]), dtype=float)
my = np.array(([1,0], [0,1]), dtype=float)


X = loadCSV('TrainDigitX.csv.gz')
Y = loadCSV("TrainDigitY.csv.gz")

Ninput = 784
Nhidden = 30
Noutput = 10

NN = Neural_Network(Ninput, Nhidden, Noutput)
m = 20
n = 3
epochs = 1

for i in range(epochs):
    NN.miniBatch(n,m,X,Y)
    NN.UpdateWeights(n, m)
    meanSquare = NN.findError(my[1], mX[1])
    meanSquare += NN.findError(my[0], mX[0])
    meanSquare = meanSquare/2
print("avg",meanSquare)

#print(NN.forward(mX[0]))
#print(NN.forward(mX[1]))
