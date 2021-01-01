import numpy as np
from matplotlib import pyplot as plt
import activation_library as ac

class Conv_Layer:
    def __init__(self, filter_size=5, filter_count=6, stride=1):
        self.size = filter_size
        self.filter_count = filter_count
        self.stride = stride
        self.filters = np.random.normal(0.0, pow(self.size, -0.5), (self.filter_count, self.size, self.size))

    def forward_propagation(self, inputs=None):
        
        #   calculate the size of the output matrix
        #   based on the formula: (x-axis - filter_size) / stride + 1
        resulting_dimension = int(  (inputs.shape[1]-self.size)/self.stride+1   )

        resulting_matrix = np.zeros((self.filter_count, resulting_dimension, resulting_dimension))

        max_stride = inputs.shape[1]-self.size

        x_min, x_max = 0, self.size
        y_min, y_max = 0, self.size

        current_filter = 0
        counter = 0
        while current_filter < self.filter_count:
            counter += 1

            dot_prod = np.sum(  
                    np.dot( self.filters[current_filter], inputs[x_min:x_max, y_min:y_max] )
                    )

            resulting_matrix[current_filter][x_min][y_min] = dot_prod
            
            if (x_min >= max_stride):
                x_max = self.size
                x_min = 0
                y_min += self.stride
                y_max += self.stride
            else:
                x_max += self.stride
                x_min += self.stride

            if (y_min > max_stride):
                current_filter += 1
                x_max = self.size
                x_min = 0
                y_max = self.size
                y_min = 0

        return resulting_matrix

    def backward_propagation(self):
        pass

class Pool_Layer:
    def __init__(self, pooling_type="max", filter_size=2, stride=2):

        if pooling_type == "avg":
            self.type_ = 1
        elif pooling_type == "max":
            self.type_ = 0

        self.size = filter_size
        self.stride = stride

    def forward_propagation(self, inputs=None):
        
        resulting_dimension = int( (inputs.shape[1] - self.size)/self.stride + 1 )

        resulting_matrix = np.zeros((inputs.shape[0], resulting_dimension, resulting_dimension))

        max_stride = inputs.shape[1] - self.size

        x_min, x_max = 0, self.size
        y_min, y_max = 0, self.size

        current_filter = 0

        while current_filter < inputs.shape[0]:
            pooling_matrix = inputs[current_filter][x_min:x_max, y_min:y_max]

            if (self.type_ == 1):
                resulting_matrix[current_filter][int(x_min/2)][int(y_min/2)] = np.sum(pooling_matrix)/pooling_matrix.size
            else:
                resulting_matrix[current_filter][int(x_min/2)][int(y_min/2)] = np.max(pooling_matrix)

            if (x_min >= max_stride):
                x_max = self.size
                x_min = 0
                y_min += self.stride
                y_max += self.stride
            else:
                x_max += self.stride
                x_min += self.stride

            if (y_min > max_stride):
                current_filter += 1
                x_max = self.size
                x_min = 0
                y_max = self.size
                y_min = 0

        return resulting_matrix
        
class Fully_Connected_Layer:
    
    def __init__(self, inputs, outputs, learning_rate):

        self.input_nodes = inputs
        self.output_nodes = outputs
        self.weights = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.output_nodes, self.input_nodes))
        self.lr = learning_rate 

    def forward_propagation(self, inputs):

        inputs_array = np.array(inputs, ndmin=2).T

        dot_product = np.dot(self.weights, inputs_array)

        return dot_product

    def backward_propagation(self, errors, outputs, inputs):
        
        errors = np.array(errors, ndmin=2).T
        outputs = np.array(outputs, ndmin=2).T
        inputs = np.array(inputs, ndmin=2).T

        weight_errors = np.dot(self.weights.T, errors)

        self.weights += self.lr * np.dot((errors * outputs * (1.0 - outputs)), np.transpose(inputs))

def main():

    training_data_file = open(r"C:\Users\ahasa\OneDrive\Asiakirjat\machine_learning\basic_network\mnist.csv")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    all_values = training_data_list[1].split(',')

    pixels = (np.asfarray(all_values[:784]) / 255 * 0.99) + 0.01
    pixels = pixels.reshape(28, 28) 

    targets = np.zeros(10) + 0.01
    targets[int(all_values[784])] = 0.99

    #   double-padding to turn the input into a 32x32 matrix
    pixels = np.pad(pixels, (2, 2), 'constant')

    # plt.imshow(pixels)
    # plt.show()

    layers = {
            "C1" : Conv_Layer(5, 6, 1),
            "P1" : Pool_Layer("avg"),
            "C2" : Conv_Layer(5, 16, 1),
            "P2" : Pool_Layer("avg"),
            "C3" : Conv_Layer(5, 120, 1),
            "FC1" : Fully_Connected_Layer(120, 84, 0.1),
            "FC2" : Fully_Connected_Layer(84, 10, 0.1)
    }

    convo_1 = layers["C1"].forward_propagation(pixels)
    convo_1 = ac.tanh(convo_1)

    pool_1 = layers["P1"].forward_propagation(convo_1)
    pool_1 = ac.tanh(pool_1)

    convo_2 = layers["C2"].forward_propagation(pool_1) 
    convo_2 = ac.tanh(convo_2)

    pool_2 = layers["P2"].forward_propagation(convo_2)
    pool_2 = ac.tanh(pool_2)

    convo_3 = layers["C3"].forward_propagation(pool_2)
    convo_3 = ac.tanh(convo_3).flatten()

    fc_1 = layers["FC1"].forward_propagation(convo_3)
    fc_1 = ac.tanh(fc_1).flatten()

    fc_2 = layers["FC2"].forward_propagation(fc_1)
    fc_2 = ac.tanh(fc_2).flatten()

    final_output = ac.softmax(fc_2)
    errors = targets - final_output

    layers["FC2"].backward_propagation(errors, final_output, fc_1)


if __name__ == '__main__':
    main()