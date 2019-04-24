from random import random
from math import exp
import copy

#initialise the nework with random weights
#Each hidden neuron will have number of input+1 * number of hiddenNuerons
#Each output neuron wil have number of hiddenNuerons+1 * number of output nuerons
def initialize_network(X, numHidden, Y):
    network = list()
    hiddenNueron = [{'weights':[random() for i in range(X + 1)]} for i in range(numHidden)]
    outNueron = [{'weights':[random() for i in range(numHidden + 1)]} for i in range(Y)]
    network.append(hiddenNueron)
    network.append(outNueron)
    return network


# Calculate the weighted sum of inputs to a neuron
#sum starts on the bias and does not iterate through it since it will always be * 1
def weightedSum(weights, inputs):
    sum = weights[-1]
    for i in range(len(weights)-1):
        sum += weights[i] * inputs[i]
    return sum

# sigmoid function
def sigmoid(s):
    return 1.0 / (1.0 + exp(-s))

# Forward propagate from the inputs to the outputs
#for each neuron sum the inputs*weights and use the sigmoid function to transform the result -1 <-> 1
def forwardPass(network, inputs):
    for layer in network:
        newInputs = []
        for neuron in layer:
            activation = round(weightedSum(neuron['weights'], inputs),8)
            neuron['output'] = sigmoid(activation)
            newInputs.append(neuron['output'])
        inputs = newInputs
    return inputs

def sigmoidPrime(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons
#iterate from the output to the input
def backPropergate(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                #error function = -(y1 - outo1) where the negative is added in the delta equation
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            #error function * sigmoid prime = delta
            neuron['delta'] = errors[j] * sigmoidPrime(neuron['output'])* (-1)

# Change in weight = delta * input
def changeInWeight(network1, X1, learningRate):
    for i in range(len(network1)):
        inputs = X1[:-1]

        #if i != 0 use the previous layers outputs as inputs
        if i != 0:
            inputs = [neuron['output'] for neuron in network1[i - 1]]
            print(inputs)
        for neuron in network1[i]:
            for j in range(len(inputs)):
                # Change in weight = delta * input

                neuron['weights'][j] = neuron['delta'] * inputs[j]
            # Change in last neuron (bias no input)
            neuron['weights'][-1] = neuron['delta']

#average the change in weights with all the weights calculated per batch
def miniBatch(m, changeInWeightsPerBatch, learningRate):
    #create m lists of change in weights
    batching = []
    for network1 in changeInWeightsPerBatch:
        newBatch = []
        for i in range(len(network1)):
            for neuron in network1[i]:
                for weight in neuron['weights']:
                    newBatch.append(weight)
        batching.append(newBatch)

    #average the change in weights for each weight * learning factor
    changeInW = []
    for j in range(len(batching[0])):
        sum = 0
        for i in range(len(batching)):
            sum += batching[i][j]
        print("this is sum", sum)
        applyLearn = (sum * learningRate) / 2
        changeInW.append(applyLearn)

    #update weight. new weight += average change in weight
    max = len(changeInW)
    counter = 0
    for i in range(len(network)):
        for neuron in network[i]:
            for n in range(len(neuron['weights'])):
                neuron['weights'][n] -= changeInW[counter]
                counter += 1


def train(X1, Y1, m, learningRate, epochs):
    #iterate through the network epoch times

    for epoch in range(epochs):
        changeInWeightsPerBatch = []
        errorSum = 0
        outputs = []
        for x in range(len(X1)):
            outputs.append(forwardPass(network, X1[x]))
            backPropergate(network, Y1[x])
            changeInWeightsPerBatch.append(copy.deepcopy(network))


        #calculate change in weights for each batch

        for layer in range(len(changeInWeightsPerBatch)):
            print(X1[layer])
            changeInWeight(changeInWeightsPerBatch[layer], X1[layer], learningRate)

        #set the weights to the new weights
        miniBatch(m, changeInWeightsPerBatch, learningRate)

        for l in range(len(Y1)):
            for o in range(len(Y1)):
                errorSum += 0.5 *(Y1[l][o] - outputs[l][o])**2
        errorSum = errorSum/2
        print(errorSum)
        print("epoch", epoch, "error", errorSum)



""" main """
network=[[  {'weights': [0.1,0.1,0.1]}, #[w1, w3, w9]
		    {'weights': [0.2,0.1,0.1]}], #[w2, w4, w10]
        [   {'weights': [0.1,0.1,0.1]},  #[w5, w7, w11]
            {'weights': [0.1,0.2,0.1]}]] #[w6, w8, w12]

X1 = [[0.1, 0.1, 1],[0.1,0.2,1]]
Y1 = [[1,0],[0,1]]
m = 2
learningRate = 0.1
epochs = 1
train(X1,Y1, m, learningRate, epochs)

for layer in network:
    print(layer)

#dont initialise network when using specific starting weights
"""
hi = initialize_network(2,2,2)
print(hi)
"""
