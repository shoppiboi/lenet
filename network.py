import numpy
import layer_library

class Network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        #   setting nodes for input-, hidden- and output layers respectively
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        self.lr = learning_rate

    def train(self, image_data, image_solution):
        image_vector = numpy.array(image_data).T
        
def main():
    net = Network(784, 30, 10, 0.01)

if __name__ == "__main__":
    main()