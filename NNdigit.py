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
        print("self z",self.z)
        self.z2 = self.sigmoid(self.z) # activation function
        print("self z",self.z2)
        self.z3 = np.dot(self.z2, self.W2)+ self.bias2 # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3) # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        z =1/(1+np.exp(-s))
        return z

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)


    def backPropLayer2(self, X, y,o, currentM):
        """ Second Layer """
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error
        tempp = []
        for i in range(len(self.z2)):
            for j in range(len(self.o_delta)):
                testval = self.z2[i] * self.o_delta[j] *-1
                tempp.append(testval)

        """ First Layer """
        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error
        tempe = []
        for i in range(len(self.z2_delta)):
            for j in range(len(X)):
                tester = self.z2_delta[i] * X[j] * -1
                tempe.append(tester)


        if currentM == 0:
            self.summedBias2 = self.o_delta *-1
            self.summedBias1 = self.z2_delta *-1
            self.summedLayer2 = tempp
            self.summedLayer1 = tempe

        else:
            self.summedBias2 += self.o_delta *-1
            self.summedBias1 +=  self.z2_delta *-1

            for k in range(len(self.summedLayer1)):
                self.summedLayer1[k] += tempe[k]

            for o in range(len(self.summedLayer2)):
                self.summedLayer2[o] += tempp[o]




    def UpdateWeights(self, n, m):
        """ update weights  """
        #update first layer of weights
        count = 0
        print("This is W1[0][0]", self.W1[0][0])
        for i in range(len(self.W1)):
            for j in range(len(self.W1[i])):
                newWeight = (self.summedLayer1[count] * n) / m
                #print("summed layer",self.summedLayer1[count], "time n",n,"/ m",m)
                #print("newWeight", newWeight)
                self.W1[i][j] -= newWeight
                count += 1
        #update second layer of weights
        count = 0
        for i in range(len(self.W2)):
            for j in range(len(self.W2[i])):
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

    def miniBatch(self,n,m,X,Y):
        count = 0
        miniBatchAmount = 0
        for i in range(len(X)):
            if count == m-1:
                count = 0
                miniBatchAmount +=1
                NN.UpdateWeights(n,m)
                print("minibatch:", miniBatchAmount)
            elif count == 5:
                sample = NN.forward(X[2])
                expected = np.argmax(Y[2])
                actual = np.argmax(sample)
                print("Actual:",actual, "Expected:", expected)

            """ forward propergate """
            resultForward = NN.forward(X[i])
            """ Back propergate through last layer """
            NN.backPropLayer2(X[i],Y[i],resultForward, i)
            #print(NN.summedLayer2)
            break
            count += 1
        print(miniBatchAmount, "mini batches ran")


    def computeLoss(Y, Yhat):
        m=Y.shape[1]
        L= -(1./m) * (np.sum(np.multiply(np.log(Yhat),Y))+np.sum(np.multiply))




#X = loadCSV('TrainDigitX.csv.gz')
X = loadCSV('TrainDigitX.csv')
Yraw = loadCSV('TrainDigitY.csv')
#Yraw = loadCSV("TrainDigitY.csv.gz")
Y=[]
for i in range(len(Yraw)):
    temp = []
    for j in range(10):
        if j == Yraw[i]:
            temp.append(0.99)
        else:
            temp.append(0.01)
    Y.append(temp)


Ninput = 784
Nhidden = 30
Noutput = 10

NN = Neural_Network(Ninput, Nhidden, Noutput)
m = 20
n = 3
epochs = 1

for i in range(epochs):
    print(i, ":")
    NN.miniBatch(n,m,X,Y)
    #NN.UpdateWeights(n, m)
    sample = NN.forward(X[2])
    expected = np.argmax(Y[2])
    actual = np.argmax(sample)

    print("Actual:",actual, "Expected:", expected)
    print()

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
"""
function to save Weights done
Quadratic cost function
Cross entropy function
Test data
"""
