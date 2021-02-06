import numpy as np

def relu(inputs):
    for x in range(inputs.shape[0]):
        inputs[x][inputs[x]<=0]=0
    return inputs

def softmax(inputs):
    e = np.exp(inputs)

    return e/e.sum()

def vals(inputs):
    e = np.exp(inputs)

    return np.log(inputs/e.sum())

def sigmoid(inputs):
    return np.array(1/(1+np.exp(-inputs)), ndmin=2)

def sigmoid_derivative(inputs):
    return np.array(inputs*(1.0 - inputs), ndmin=2)

def tanh(inputs):
    return np.tanh(inputs) 