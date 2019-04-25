import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.array(([0.1, 0.1],[0.2, 0.1]), dtype=float)
        print("weights1", self.weights1)
        self.weights2   = np.array(([0.1, 0.1],[0.1, 0.2]), dtype=float)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1)+0.1)
        self.output = sigmoid(np.dot(self.layer1, self.weights2)+0.1)
        print(nn.output)

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        print(d_weights2)
        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    X = np.array(([0.1, 0.1]), dtype=float)
    y = np.array(([1,0]), dtype=float)
    nn = NeuralNetwork(X,y)

    for i in range(1):
        nn.feedforward()
        nn.backprop()
