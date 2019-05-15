from random import random
from math import exp
import copy
import csv
import numpy as np


class Neural_Network(object):
    def __init__(self, input, hidden, output):
        #parameters
        self.inputSize = input
        self.hiddenSize = hidden
        self.outputSize = output

        #weights and bias initialization
        self.W1 = np.array(([0.1, 0.1],[0.2, 0.1]), dtype=float) #[hidden[input]]
        self.W2 = np.array(([0.1, 0.1],[0.1, 0.2]), dtype=float) #[output[hidden]]
        self.bias1 = np.array(([0.1,0.1]), dtype=float) #size of hidden
        self.bias2 = np.array(([0.1,0.1]), dtype=float) #size of output

    def findError(self, y, X):
        o = NN.forwardFeed(X)
        meanSquare1 = 0.5*(y[0] -  o[0])**2
        meanSquare2 =0.5*(y[1] -  o[1])**2
        return meanSquare1 + meanSquare2

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def forwardFeed(self, X):
        self.hiddenNet = np.zeros(self.hiddenSize)
        self.hiddenOut = np.zeros(self.hiddenSize)
        self.outputNet = np.zeros(self.outputSize)
        self.outputOut = np.zeros(self.outputSize)
        #forward feed through first layer
        for i in range(len(self.W1)):
            for j in range(len(self.W1[i])):
                self.hiddenNet[i] += X[j] * self.W1[i][j]
            self.hiddenNet[i] += self.bias1[i]
        #sigmoid each value
        for i in range(len(self.hiddenOut)):
            self.hiddenOut[i] = self.sigmoid(self.hiddenNet[i])
        #forward feed through second layer
        for i in range(len(self.W2)):
            for j in range(len(self.W2[i])):
                self.outputNet[i] += self.hiddenOut[j] * self.W2[i][j]
            self.outputNet[i] += self.bias2[i]
        #sigmoid each value
        for i in range(len(self.outputOut)):
            self.outputOut[i] = self.sigmoid(self.outputNet[i])

        return self.outputOut

    #if errors check the positioning of errorToNode
    #somehow rework errortonode initialisation
    def backPropagate(self, X,y,currentM):
        """ Second Layer """
        self.delta = np.zeros(self.outputSize)
        for i in range(self.outputSize):
            error = (y[i]-self.outputOut[i])
            self.delta[i] = self.sigmoidPrime(self.outputOut[i]) * error# *-1

        self.errorToNode = np.random.randn(self.outputSize, self.hiddenSize) #delta will have same len as output
        # final error
        for i in range(len(self.delta)):
            for j in range(self.hiddenSize):
                self.errorToNode[i][j] = self.delta[i]*self.hiddenOut[j]*-1

        """ First Layer """

        errorCausedByHiddenNode = np.random.randn(self.outputSize, self.hiddenSize)
        #finding error caused by each weight
        for i in range(len(self.W2)):
            for j in range(len(self.W2[i])):
                errorCausedByHiddenNode[i][j] = self.delta[i]*self.W2[i][j] #*-1

        #summing the errors from each hidden node
        error2 = np.zeros(self.hiddenSize)
        for i in range(len(errorCausedByHiddenNode)):
            for j in range(len(errorCausedByHiddenNode[0])):
                error2[j] += errorCausedByHiddenNode[i][j]

        #setting delta
        self.delta2 = np.zeros(self.hiddenSize)
        for i in range(self.hiddenSize):
            self.delta2[i]= error2[i]*self.sigmoidPrime(self.hiddenOut[i]) #*-1


        #final error
        self.errorToNode2 =  np.random.randn( self.hiddenSize,self.inputSize)
        for i in range(len(self.delta2)):
            for j in range(self.inputSize):
                self.errorToNode2[i][j] = self.delta2[i]*X[j]*-1


        if currentM == 0:
            self.summedBias2 = self.delta *-1
            self.summedBias1 = self.delta2 *-1
            self.summedLayer2 = self.errorToNode
            self.summedLayer1 = self.errorToNode2
        else:
            for i in range(len(self.errorToNode)):
                for j in range(len(self.errorToNode[i])):
                    self.summedLayer2[i][j] += self.errorToNode[i][j]
            for i in range(len(self.errorToNode2)):
                for j in range(len(self.errorToNode2[i])):
                    self.summedLayer1[i][j] += self.errorToNode2[i][j]
            self.summedBias2 += self.delta *-1
            self.summedBias1 +=  self.delta2 *-1



    def UpdateWeights(self, n, m):
        """ update weights  """
        #update first layer of weights
        for i in range(len(self.W1)):
            for j in range(len(self.W1[i])):
                newWeight = (self.summedLayer1[i][j] * n) / m
                self.W1[i][j] -= newWeight

        #update second layer of weights
        for i in range(len(self.W2)):
            for j in range(len(self.W2[i])):
                newWeight = (self.summedLayer2[i][j] * n) / m
                self.W2[i][j] -= newWeight
        #Update Bias
        for i in range(len(self.bias1)):
            newWeight = (self.summedBias1[i] * n) / m
            self.bias1[i] -= newWeight

        for i in range(len(self.bias2)):
            newWeight = (self.summedBias2[i] * n) / m
            self.bias2[i] -= newWeight



    def miniBatch(self,n,m,X,Y):
        count = 0
        miniBatchAmount = 0
        for i in range(len(X)):
            if i == m-1:
                count = 0
                miniBatchAmount +=1
                self.UpdateWeights(n,m)
                self.resetBatchVal()

            resultForward = self.forwardFeed(X[i])
            self.backPropagate(X[i],Y[i], i)
            count += 1


    # reset summing values within back propergation function
    def resetBatchVal(self):
        for i in range(len(self.summedLayer2)):
            for j in range(len(self.summedLayer2[i])):
                self.summedLayer2[i][j] =0
        for i in range(len(self.summedLayer1)):
            for j in range(len(self.summedLayer1[i])):
                self.summedLayer1[i][j] =0
        for i in range(len(self.bias1)):
            self.bias1[i] = 0
        for i in range(len(self.bias2)):
            self.bias2[i] = 0

    def trainNetwork(self, X, Y, n, m, epochs):
        for i in range(epochs):
            print()
            print("Epoch",i)
            self.miniBatch(n,m,X,Y)

            print("---------------------------------------------------------")
            for i in range(len(X)):
                sample = self.forwardFeed(X[i])
                expected = Y[i]
            print("Actual output:",sample, "Expected output:", expected)


    def printBefore(self):
        meanSquare = self.findError(my[1], mX[1])
        meanSquare += self.findError(my[0], mX[0])
        meanSquare = meanSquare/2
        print()
        print("Mean Squared Error:")
        print("------------------------------")
        print("Before: ", meanSquare)

        print()
        print("Weights")
        print("------------------------------")
        count = 0
        for i in range(len(self.W1)):
            for j in range(len(self.W1)):
                print("Weight", count+1, ":", self.W1[i][j])
                count+=1

        count = 0
        for i in range(len(self.W2)):
            for j in range(len(self.W2)):
                print("Weight", count+5, ":", self.W2[i][j])
                count+=1

        print()
        p = NN.forwardFeed(mX[0])
        print("Input: 1")
        print("------------------------------")
        for i in range(len(p)):
            print("Output", i+1, ":",p[i])
        p = NN.forwardFeed(mX[1])
        print()
        print("Input: 2")
        print("------------------------------")
        for i in range(len(p)):
            print("Output", i+1, ":",p[i])

    def printAfter(self):
        print()
        print("Weights-numbering incorrect")
        print("------------------------------")
        count = 0
        for i in range(len(self.W1)):
            for j in range(len(self.W1)):
                print("Weight", count+1, ":", self.W1[i][j])
                count+=1

        count = 0
        for i in range(len(self.W2)):
            for j in range(len(self.W2)):
                print("Weight", count+5, ":", self.W2[i][j])
                count+=1

        meanSquare = self.findError(my[1], mX[1])
        meanSquare += self.findError(my[0], mX[0])
        meanSquare = meanSquare/2
        print()
        print("Mean Squared Error:")
        print("------------------------------")
        print("After: ", meanSquare)
        print()
        print("Hidden Bias Weights:")
        print("------------------------------")
        for i in range(len(self.bias1)):
            print("Weight", i+9, ":", self.bias1[i])

        print()
        print("Output Bias Weights:")
        print("------------------------------")
        for i in range(len(self.bias2)):
            print("Weight", i+11, ":", self.bias2[i])


""" Driver """

mX = np.array(([0.1, 0.1],[0.1,0.2]), dtype=float)
my = np.array(([1,0], [0,1]), dtype=float)
n=0.1
m=2
epochs = 1
NN = Neural_Network(2,2,2)
NN.printBefore()
NN.miniBatch(n,m,mX,my)
NN.UpdateWeights(n, m)
NN.printAfter()

#NN.trainNetwork( mX, my, n, m, epochs)
"""
print()
print(NN.W1)
print(NN.W2)
print(NN.bias1)
print(NN.bias2)
print(NN.forwardFeed(mX[0]))
print(NN.forwardFeed(mX[1]))
"""
