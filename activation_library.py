import numpy as np

#   separate library for all potential activation functions I may wish to try

def activation_ReLu(inputs):
    for x in range(inputs.shape[0]):
        inputs[x][inputs[x]<=0]=0
    return inputs