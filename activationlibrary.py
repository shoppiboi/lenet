import numpy as np

def relu(inputs):
    for x in range(inputs.shape[0]):
        inputs[x][inputs[x]<=0]=0
    return inputs

def softmax(inputs):
    e = np.exp(inputs)

    return e/e.sum()

def sigmoid(inputs):
    return 1/(1+np.exp(-inputs))

def tanh(inputs):
    return np.tanh(inputs)