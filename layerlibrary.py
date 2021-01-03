import numpy as np
from matplotlib import pyplot as plt
import activationlibrary as ac

class StrideLayer:

    def __init__(self, kernel_size, stride):
        self.size = kernel_size
        self.stride = stride

    #   returns a tuple of adjusted values to proceed with either convolution or pooling
    def increment_indexes(self, x_min, x_max, y_min, y_max, current_filter):  

        #   if the maximum stride distance has been reached horizontally
        if (x_min >= self.max_stride):
            x_max = self.size
            x_min = 0
            y_min += self.stride
            y_max += self.stride
        else:
            x_max += self.stride
            x_min += self.stride

        #   if the maximum stride distance has been reached vertically
        if (y_min > self.max_stride):
            current_filter += 1
            x_max = self.size
            x_min = 0
            y_max = self.size
            y_min = 0

        return x_min, x_max, y_min, y_max, current_filter

    def return_resulting_matrix(self, input_size, z):

        #   calculate the size of the output matrix
        #   based on the formula: (x-axis - filter_size) / stride + 1
        resulting_dimension = int(  (input_size-self.size)/self.stride+1   )

        #   the resulting matrix is of matrix-size (z, x, y)
        return np.zeros((z, resulting_dimension, resulting_dimension))

    def set_max_stride(self, input_x):
        self.max_stride = input_x-self.size

class Convolutional(StrideLayer):
    
    def __init__(self, kernel_size=5, filter_count=6, variance_root=32, stride=1):
        StrideLayer.__init__(self, kernel_size, stride)

        self.filter_count = filter_count
        weight_dimension = (self.filter_count, self.size, self.size)
        
        #   "He - initialization" for the weights root(2/variance)
        self.filters = np.random.normal(0.0, pow(2/variance_root, -0.5), (weight_dimension))

    def forward_propagation(self, inputs=None):
    
        #   depth (z) of resulting matrix is number of filters
        resulting_matrix = StrideLayer.return_resulting_matrix(self, 
                                inputs.shape[1], 
                                self.filter_count
                            )

        StrideLayer.set_max_stride(self, inputs.shape[1])

        x_min, x_max = 0, self.size
        y_min, y_max = 0, self.size

        current_filter = 0
        while current_filter < self.filter_count:
            
            temp_arr = []
            if (len(inputs.shape) > 2):
                for i in range(inputs.shape[0] - 1):
                    dot_prod = np.dot( self.filters[current_filter], inputs[i][x_min:x_max, y_min:y_max] )
                    sum_ = np.sum(dot_prod)
                    temp_arr.append(sum_)
            else:
                dot_prod = np.dot( self.filters[current_filter], inputs[x_min:x_max, y_min:y_max] )
                temp_arr.append(np.sum(dot_prod))

            #   replace 0 at given (z, x, y) position with the product
            resulting_matrix[current_filter][x_min][y_min] = np.sum(temp_arr)

            
            #   values adjusted for next while-loop iteration
            x_min, x_max, y_min, y_max, current_filter = StrideLayer.increment_indexes(self, 
                                                            *(x_min, x_max, y_min, y_max, current_filter),
                                                        )

        return resulting_matrix

    def backward_propagation(self):
        pass

class Pooling(StrideLayer):

    def __init__(self, pooling_type="max", kernel_size=2, stride=2):
        StrideLayer.__init__(self, kernel_size, stride)

        #   assign the lambda function for pooling based on pooling_type given
        if pooling_type == "avg":
            self.pool = lambda a : np.sum(a)/a.size
        elif pooling_type == "max":
            self.pool = lambda b : np.max(b)


    def forward_propagation(self, inputs=None):
        
        #   unlike Convolutional;
        #   depth (z) of resulting matrix is depth of input matrix  
        resulting_matrix = StrideLayer.return_resulting_matrix(self,
                                inputs.shape[1], 
                                inputs.shape[0]
                            )

        StrideLayer.set_max_stride(self, inputs.shape[1])

        x_min, x_max = 0, self.size
        y_min, y_max = 0, self.size

        current_filter = 0

        while current_filter < inputs.shape[0]:
            pooling_matrix = inputs[current_filter][x_min:x_max, y_min:y_max]

            #   applies the pre-determined pooling function to the convolutions
            resulting_matrix[current_filter][int(x_min/2)][int(y_min/2)] = self.pool(pooling_matrix)

            x_min, x_max, y_min, y_max, current_filter = StrideLayer.increment_indexes(self, 
                                                            *(x_min, x_max, y_min, y_max, current_filter)
                                                        )

        return resulting_matrix

class FullyConnectedLayer:
    
    def __init__(self, inputs, outputs):

        self.input_nodes = inputs
        self.output_nodes = outputs

        #   "He initialization" for the weights
        self.weights = np.random.normal(0.0, pow(2/self.input_nodes, -0.5), (self.output_nodes, self.input_nodes))

    def forward_propagation(self, inputs):

        dot_product = np.dot(self.weights, inputs)

        return dot_product

    def back_propagation(self, errors, outputs, inputs):

        self.weights += LEARNING_RATE * np.dot((errors * outputs * (1.0 - outputs)), inputs.T)

LEARNING_RATE = 0.03
BIAS = 0

#   for testing purposes
#   plots the given matrix so I can see what each
#   feature map looks like
def display_feature_maps(inputs):
    for x in range(inputs.shape[0]):
        plt.imshow(inputs[x])
        plt.show()

def return_cross_entropy(target, outputs):
    actual_output = np.zeros(outputs.size) + 0.01

    actual_output[int(target)] = 0.99

    temp_arr = []

    for x in range(actual_output.size):
        temp_arr.append(
                - ((actual_output[x] * np.log10(outputs[x])) + ((1 - actual_output[x]) * np.log10(1 - outputs[x])))
        )

    return np.sum(temp_arr)


##################
#   the main() function will be run on the network.py script

#   Everything is here for now merely for efficiency's sake, but
#   this script will be imported as a library in the network.py script
#   just as how 'activationalibrary' has been imported in this script
##################

def main():

    training_data_file = open(r"C:\Users\ahasa\OneDrive\Asiakirjat\machine_learning\basic_network\mnist.csv")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    #   split the columns on the .csv file by commas
    all_values = training_data_list[1].split(',')

    #   reducing all values within the range of 0.01 and 1.0 to make the image grayscale
    pixels = (np.asfarray(all_values[:784]) / 255 * 0.99) + 0.01
    pixels = pixels.reshape(28, 28) 

    targets = np.zeros(10)
    targets[int(all_values[784])] = 1
    targets = np.array(targets, ndmin=2).T

    #   double-padding to turn the input into a 32x32 matrix
    pixels = np.pad(pixels, (2, 2), 'constant')

    #   create a dictionary of layers with their respective parameters
    layers = {
            "C1" : Convolutional(5, 6, 32),
            "P1" : Pooling("max"),
            "C2" : Convolutional(5, 16, 28),
            "P2" : Pooling("max"),
            "C3" : Convolutional(5, 120, 14),
            "FC1" : FullyConnectedLayer(120, 84),
            "FC2" : FullyConnectedLayer(84, 10) 
    }

    #   apply forward propagation for all layers in the dictionary
    out_cv1 = layers["C1"].forward_propagation(pixels)
    out_cv1 = ac.tanh(out_cv1)

    out_pool1 = layers["P1"].forward_propagation(out_cv1)
    out_pool1 = ac.tanh(out_pool1)

    out_cv2 = layers["C2"].forward_propagation(out_pool1) 
    out_cv2 = ac.tanh(out_cv2)

    out_pool2 = layers["P2"].forward_propagation(out_cv2)
    out_pool2 = ac.tanh(out_pool2)

    out_cv3 = layers["C3"].forward_propagation(out_pool2)
    out_cv3 = ac.tanh(out_cv3).flatten()

    out_cv3 = np.array(out_cv3, ndmin=2).T
    out_fc1 = layers["FC1"].forward_propagation(out_cv3)
    out_fc1 = ac.tanh(out_fc1).flatten()

    out_fc1 = np.array(out_fc1, ndmin=2).T
    out_fc2 = layers["FC2"].forward_propagation(out_fc1)

    final_output = ac.softmax(out_fc2)

    #   apply back propagation for all layers in the dictionary
    output_errors = targets - final_output

    err_fc1 = np.dot(layers["FC2"].weights.T, output_errors)

    layers["FC2"].back_propagation(output_errors, final_output, out_fc1)

    layers["FC1"].back_propagation(err_fc1, out_fc1, out_cv3)

if __name__ == '__main__':
    main()