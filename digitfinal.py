from random import random
from math import exp
import copy
import csv
import numpy as np

def loadCSV(fileName):
    data = np.loadtxt(fileName, dtype=float, delimiter=",")
    print(fileName, " loaded in successfuly")
    return data

class Neural_Network(object):
    def __init__(self, input, hidden, output):
        #parameters
        self.inputSize = input
        self.hiddenSize = hidden
        self.outputSize = output

        #weights and bias initialization
        """
        self.W1 = np.random.randn(self.hiddenSize, self.inputSize )
        self.W2 = np.random.randn(self.outputSize, self.hiddenSize)
        self.bias1 = np.random.randn(self.hiddenSize)
        self.bias2 = np.random.randn(self.outputSize)
        """
        #testcase
        self.W1 = np.array(([0.1, 0.1],[0.2, 0.1]), dtype=float) #[hidden[input]]
        self.W2 = np.array(([0.1, 0.1],[0.1, 0.2]), dtype=float) #[output[hidden]]
        self.bias1 = np.array(([0.1,0.1]), dtype=float) #size of hidden
        self.bias2 = np.array(([0.1,0.1]), dtype=float) #size of output

        #test cases to check size of data structures
        print(self.W1)
        print(len(self.W1), "should equal", hidden)
        print(len(self.W1[0]), "should equal", input)
        print(len(self.W2), "should equal", output)
        print(len(self.W2[0]), "should equal", hidden)
        print(len(self.bias1),"should equal",hidden)
        print(len(self.bias2),"should equal",output)



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

        #return o

    #if errors check the positioning of errorToNode
    #somehow rework errortonode initialisation
    def backPropagate(self, X,y,currentM):
        """ Second Layer """
        self.delta = np.zeros(self.outputSize)
        for i in range(self.outputSize):
            error = (y[i]-self.outputOut[i])
            self.delta[i] = self.sigmoidPrime(self.outputOut[i]) * error# *-1

        print("delta", self.delta)
        self.errorToNode = np.random.randn(self.outputSize, self.hiddenSize) #delta will have same len as output
        # final error
        for i in range(len(self.delta)):
            for j in range(self.hiddenSize):
                self.errorToNode[i][j] = self.delta[i]*self.hiddenOut[j]*-1
        print("errorToNode",self.errorToNode)

        """ First Layer """

        errorCausedByHiddenNode = np.random.randn(self.outputSize, self.hiddenSize)
        error2 = np.zeros(self.hiddenSize)

        #finding error caused by each weight
        for i in range(len(self.W1)):
            for j in range(len(self.W1[i])):
                errorCausedByHiddenNode[i][j] = self.delta[i]*self.W2[i][j] #*-1

        #summing the errors from each hidden node
        for i in range(len(errorCausedByHiddenNode)):
            for j in range(len(errorCausedByHiddenNode[i])):
                error2[i] += errorCausedByHiddenNode[j][i]
        print("error2",error2)
        print()

        #setting delta
        self.delta2 = np.zeros(self.hiddenSize)
        for i in range(self.hiddenSize):
            self.delta2[i]= error2[i]*self.sigmoidPrime(self.hiddenOut[i]) #*-1
        print("delta 2",self.delta2)
        print()

        #final error
        self.errorToNode2 =  np.random.randn( self.hiddenSize,self.inputSize)
        for i in range(len(self.delta2)):
            for j in range(self.inputSize):
                self.errorToNode2[i][j] = self.delta2[i]*X[j]*-1
        print("errorToNode2",self.errorToNode2)

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

    def miniBatch(self,n,m,mX,my):
        for i in range(m):
            """ forward propergate """
            resultForward = NN.forwardFeed(mX[i])
            """ Back propergate through last layer """
            print("here")
            NN.backPropagate(mX[i],my[i], i)




mX = np.array(([0.1, 0.1],[0.1,0.2]), dtype=float)
my = np.array(([1,0], [0,1]), dtype=float)
n=0.1
m=2
NN = Neural_Network(2,2,2)
NN.miniBatch(n,m,mX,my)
NN.UpdateWeights(n, m)
print(NN.W1)
print(NN.W2)
"""
NN.forwardFeed(mX[0])
print(NN.outputOut, "Should equal 0.5513784696896066, 0.5644490253545453" )
NN.backPropergate(mX[0],my[0])
NN.forwardFeed(mX[1])
print(NN.outputOut, "Should equal 0.5515631366563031,  0.5646937754140904")
"""


"""
input = 3
hidden = 2
output = 4

NN = Neural_Network(input,hidden,output)
"""
