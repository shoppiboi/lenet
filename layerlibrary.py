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
    
    def __init__(self, kernel_size=5, filter_count=6, stride=1):
        StrideLayer.__init__(self, kernel_size, stride)

        self.filter_count = filter_count
        weight_dimension = (self.filter_count, self.size, self.size)
        self.filters = np.random.normal(0.0, pow(self.size, -0.5), (weight_dimension))

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

            dot_prod = np.sum(  
                    np.dot( self.filters[current_filter], inputs[x_min:x_max, y_min:y_max] )
                    )

            #   replace 0 at given (z, x, y) position with the product
            resulting_matrix[current_filter][x_min][y_min] = dot_prod
            
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

class FullyConnected:
    
    def __init__(self, inputs, outputs, learning_rate):

        self.input_nodes = inputs
        self.output_nodes = outputs
        self.weights = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.output_nodes, self.input_nodes))
        self.lr = learning_rate 

    def forward_propagation(self, inputs):

        inputs_array = np.array(inputs, ndmin=2).T

        dot_product = np.dot(self.weights, inputs_array)

        return dot_product

    def back_propagation(self, errors, outputs, inputs):
        
        errors = np.array(errors, ndmin=2).T
        outputs = np.array(outputs, ndmin=2).T
        inputs = np.array(inputs, ndmin=2).T

        self.weights += self.lr * np.dot((errors * outputs * (1.0 - outputs)), np.transpose(inputs))


LEARNING_RATE = 0.1

#   for testing purposes
#   plots the given matrix so I can see what each
#   feature map looks like
def display_feature_maps(inputs):
    for x in range(inputs.shape[0]):
        plt.imshow(inputs[x])
        plt.show()

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

    #   create a dictionary of layers with their respective parameters
    layers = {
            "C1" : Convolutional(5, 6, 1),
            "P1" : Pooling("avg"),
            "C2" : Convolutional(5, 16, 1),
            "P2" : Pooling("avg"),
            "C3" : Convolutional(5, 120, 1),
            "FC1" : FullyConnected(120, 84, LEARNING_RATE),
            "FC2" : FullyConnected(84, 10, LEARNING_RATE)
    }

    #   apply forward propagation for all layers in the dictionary
    out_cv1 = layers["C1"].forward_propagation(pixels)
    out_cv1 = ac.relu(out_cv1)

    out_pool1 = layers["P1"].forward_propagation(out_cv1)
    out_pool1 = ac.relu(out_pool1)

    out_cv2 = layers["C2"].forward_propagation(out_pool1) 
    out_cv2 = ac.relu(out_cv2)

    out_pool2 = layers["P2"].forward_propagation(out_cv2)
    out_pool2 = ac.relu(out_pool2)

    out_cv3 = layers["C3"].forward_propagation(out_pool2)
    out_cv3 = ac.relu(out_cv3).flatten()

    out_fc1 = layers["FC1"].forward_propagation(out_cv3)
    out_fc1 = ac.relu(out_fc1).flatten()

    out_fc2 = layers["FC2"].forward_propagation(out_fc1)
    out_fc2 = ac.relu(out_fc2).flatten()

    final_output = ac.softmax(out_fc2)

    #   apply back propagation for all layers in the dictionary
    err_final_output = targets - final_output

    layers["FC2"].back_propagation(err_final_output, final_output, out_fc1)

    err_fc1 = np.dot(layers["FC2"].weights.T, out_fc2)
    layers["FC1"].back_propagation(err_fc1, out_fc1, out_cv3)

if __name__ == '__main__':
    main()