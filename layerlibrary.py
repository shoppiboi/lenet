import numpy as np

############ CONSTANTS
LEARNING_RATE = 0.03
BIAS = 0
############ CONSTANTS

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

    #   determine the maximum stride distance on the filter and store it as a variabe
    def set_max_stride(self, input_x):
        self.max_stride = input_x-self.size

class Convolutional(StrideLayer):
    
    def __init__(self, kernel_size=5, filter_count=6, variance_root=32, stride=1, learning_rate=LEARNING_RATE):
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
            #   if the input is three dimensional
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
                                                            *(x_min, x_max, y_min, y_max, current_filter),  #   tuple
                                                        )

        return resulting_matrix

    def backward_propagation(self):
        pass

class Pooling(StrideLayer):

    def __init__(self, pooling_type="max", kernel_size=2, stride=2, learning_rate=LEARNING_RATE):
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
                                                            *(x_min, x_max, y_min, y_max, current_filter)   #   tuple
                                                        )

        return resulting_matrix

class FullyConnectedLayer:
    
    def __init__(self, inputs, outputs, learning_rate=LEARNING_RATE):

        self.input_nodes = inputs
        self.output_nodes = outputs

        #   "He initialization" for the weights
        self.weights = np.random.normal(0.0, pow(2/self.input_nodes, -0.5), (self.output_nodes, self.input_nodes))

    def forward_propagation(self, inputs):

        dot_product = np.dot(self.weights, inputs)

        return dot_product

    def back_propagation(self, error_derivative):

        # print(error_derivative.shape)

        self.weights += LEARNING_RATE * error_derivative

def return_cross_entropy(target, outputs):
    expected_output = np.zeros(outputs.size)

    print(expected_output.shape)

    expected_output[target] = 1

    return np.sum(-((expected_output * np.log10(outputs)) + ((1 - expected_output) * np.log10(1 - outputs))))

def return_derivative_cross_entropy(target, outputs):
    expected_output = np.zeros(outputs.size)
    expected_output = np.reshape(outputs.size, 1)

    #   set expected output as 1
    expected_output[int(target)] = 1

    #   same formula as for normal cross entropy, except output values have been inversed
    return np.array(-((expected_output * pow(outputs, -1)) + ((1 - expected_output) * pow((1 - outputs), -1))))



#   returns the derivative of the softmax_inputs
#   derivative of softmax layer with respect to output layer input
#   e^[0] * (e^[1] + e^[2] + ... + e^[n]) / (e^[0]+e^[1]+e^[2]+...+e^[n])^2
def softmax_derivative(inputs):

    #   creates a numpy array of exponentials to the power of all inputs
    e_ = np.exp(inputs)
    output = []

    derivative = lambda a, b, c : (a * (np.sum(b))) / pow(np.sum(c), 2)

    for x in range(e_.size):
        #   append derivatives to the der_ array
        output.append(derivative(e_[x], #   current element
                        [y for y in e_ if y != e_[x]],  #   all elements excluding current element
                        e_) #   all elements
                        )

    return np.array(output)
    
def return_derivative_input_output_weights(inputs, shape):

    temp = np.tile(inputs, shape)

    # print(temp)

    return temp