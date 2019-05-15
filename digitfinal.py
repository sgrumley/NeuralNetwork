from random import random
from math import exp
import copy
import csv
import numpy as np
import sys

#function to read data into np arrays
def loadCSV(fileName):
    data = np.loadtxt(fileName, dtype=float, delimiter=",")
    print(fileName, " loaded in successfuly")
    return data

# pre process Y labels into comparible data 3 = [0,0,0,1,0,0,0,0,0,0]
def hotOne(Yraw):
    Y=[]
    for i in range(len(Yraw)):
        temp = []
        for j in range(10):
            if j == Yraw[i]:
                temp.append(1.0)
            else:
                temp.append(0.0)
        Y.append(temp)
    return Y

# function to read weights and biases in from a txt file
def readWeights():
    NN =Neural_Network(input,hidden,output)
    f = open("W1.txt","r")
    i=0
    for line in f:
        j=0
        for word in line.split():
            NN.W1[i][j] = word
            j+=1
        i+=1
    f.close()

    file = open("W2.txt","r")
    i=0
    for line in file:
        j=0
        for word in line.split():
            NN.W2[i][j] = word
            j+=1
        i+=1
    file.close()
    return NN

# function for saving the weights and biases to a txt file
def writeWeights(NN):
    f= open("W1.txt","w+")
    for i in range(len(NN.W1)):
        for j in range(len(NN.W1[i])):
            f.write(str(NN.W1[i][j])+ " ")
        f.write("\n")
    f.close
    file = open("W2.txt","w+")
    for i in range(len(NN.W2)):
        for j in range(len(NN.W2[i])):
            file.write(str(NN.W2[i][j])+ " ")
        file.write("\n")
    file.close

    file = open("B1.txt","w+")
    for i in range(len(NN.bias1)):
        file.write(str(NN.bias1[i])+ " ")
    file.close

    file = open("B2.txt","w+")
    for i in range(len(NN.bias2)):
        file.write(str(NN.bias2[i])+ " ")
    file.close



class Neural_Network(object):
    def __init__(self, input, hidden, output):
        #parameters
        self.inputSize = input
        self.hiddenSize = hidden
        self.outputSize = output

        #weights and bias initialization
        self.W1 = np.random.randn(self.hiddenSize, self.inputSize )
        self.W2 = np.random.randn(self.outputSize, self.hiddenSize)
        self.bias1 = np.random.randn(self.hiddenSize)
        self.bias2 = np.random.randn(self.outputSize)

    # actrivation function
    def sigmoid(self, s):
        #running into issues with large numbers being passed in
        # Let them be 1 instead of breaking
        try:
            z =1/(1+ np.exp(-s))
        except:
            print("stil happening bruh")
            s = float('inf')
            z =1/(1+ np.exp(-s))

        return z

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    # Forward feed through the neural net
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


    # find the error of each weight
    def backPropagate(self, X,y,currentM):
        """ Second Layer """
        self.delta = np.zeros(self.outputSize)
        # delta =  sigmoid * error
        for i in range(self.outputSize):
            error = (y[i]-self.outputOut[i])
            self.delta[i] = self.sigmoidPrime(self.outputOut[i]) * error

        self.errorToNode = np.random.rand(self.outputSize, self.hiddenSize)

        # Error to the node = delta * hidden layer output
        for i in range(len(self.delta)):
            for j in range(self.hiddenSize):
                self.errorToNode[i][j] = self.delta[i]*self.hiddenOut[j]*-1

        """ First Layer """
        errorCausedByHiddenNode = np.random.rand(self.outputSize, self.hiddenSize)
        #finding error caused by each weight
        # delta * weight
        for i in range(len(self.W2)):
            for j in range(len(self.W2[i])):
                errorCausedByHiddenNode[i][j] = self.delta[i]*self.W2[i][j]

        #summing the errors to find error caused by each hidden node
        #eg w5 + w6 going to different output node
        error2 = np.zeros(self.hiddenSize)
        for i in range(len(errorCausedByHiddenNode)):
            for j in range(len(errorCausedByHiddenNode[0])):
                error2[j] += errorCausedByHiddenNode[i][j]

        #setting delta
        #error * sigmoid derivative of hidden node outputs
        self.delta2 = np.zeros(self.hiddenSize)
        for i in range(self.hiddenSize):
            self.delta2[i]= error2[i]*self.sigmoidPrime(self.hiddenOut[i])

        #Error caused by each weight of layer 1
        self.errorToNode2 =  np.random.rand( self.hiddenSize,self.inputSize)
        for i in range(len(self.delta2)):
            for j in range(self.inputSize):
                self.errorToNode2[i][j] = self.delta2[i]*X[j]*-1

        # if its the first forward feed set values
        # else sum error of weights
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

    # Update the weights with the sum of errors
    def UpdateWeights(self, n, m):
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
        #Update Bias layer 1
        for i in range(len(self.bias1)):
            newWeight = (self.summedBias1[i] * n) / m
            self.bias1[i] -= newWeight
        #Update Bias layer 1
        for i in range(len(self.bias2)):
            newWeight = (self.summedBias2[i] * n) / m
            self.bias2[i] -= newWeight

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

    # control a full interation through all the data and update the weights every m iterations
    def miniBatch(self,n,m,X,Y):
        count = 0
        miniBatchAmount = 0
        #for i in range(len(X)):
        for i in range(2000):
            if count == m-1:
                count = 0
                miniBatchAmount +=1
                self.UpdateWeights(n,m)
                self.resetBatchVal()
            # every 500 iterations print to console to show the program is not stuck
            if i % 500 == 0:
                print("progressing",i, "out of ", len(X))

            resultForward = self.forwardFeed(X[i])
            self.backPropagate(X[i],Y[i], i)
            count += 1
        print(miniBatchAmount, "mini batches ran")


    def trainNetwork(self, X, Y, testX, testY, n, m, epochs):
        for i in range(epochs):
            print()
            print("Epoch",i)
            self.miniBatch(n,m,X,Y)
            sample = self.forwardFeed(X[2])
            correct = 0
            incorrect = 0
            print("---------------------------------------------------------")
            for i in range(len(testX)):
                sample = self.forwardFeed(testX[i])
                expected = np.argmax(testY[i])
                actual = np.argmax(sample)
                if expected == actual:
                    correct+=1
                else:
                    incorrect+=1

            print("Correct:",correct, "Incorrect:", incorrect)
            total = correct+incorrect
            print("Total tests:", total)
            print("Accuracy:", correct/total)


""" driver """
# cmd line to run program as intended
#python digitfinal.py 784 30 10 TrainDigitX.csv.gz TrainDigitY.csv.gz TestDigitX.csv.gz TestDigitY.csv.gz
# set parameters from commandline arguments
input = int(sys.argv[1])
hidden = int(sys.argv[2])
output = int(sys.argv[3])
filenameX = sys.argv[4]
filenameY = sys.argv[5]
filenameTestX = sys.argv[6]
filenameTestY = sys.argv[7]

# Read in inputs and expected outputs from file
X = loadCSV(filenameX)
Yraw = loadCSV(filenameY)
testX = loadCSV(filenameTestX)
testYraw =  loadCSV(filenameTestY)

# process output labels to match forward feed output
Y=hotOne(Yraw)
testY = hotOne(testYraw)

# parameters to assist learning efficiency
n=3.0
m=20
epochs = 30
# Create Neural Network
NN = Neural_Network(input,hidden,output)
NN.trainNetwork(X, Y, testX, testY, n, m, epochs)
writeWeights(NN)


#readWeights()
