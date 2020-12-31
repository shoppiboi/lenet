import numpy as np
from matplotlib import pyplot as plt

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

        max_stride = inputs.shape[0]-self.size

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

    def activation_ReLu(self, inputs):
        for x in range(inputs.shape[0]):
            inputs[x][inputs[x]<=0]=0

        return inputs

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

        x_min, x_max = 0, 2
        y_min, y_max = 0, 2

        current_filter = 0

        while current_filter < inputs.shape[0]:
            pooling_matrix = inputs[current_filter][x_min:x_max, y_min:y_max]

            if (self.type_ == 1):
                resulting_matrix[current_filter][int(x_min/2)][int(y_min/2)] = np.sum(pooling_matrix)/pooling_matrix.size
            else:
                resulting_matrix[current_filter][int(x_min/2)][int(y_min/2)] = np.max(pooling_matrix)

            if (x_min >= max_stride):
                x_max = 2
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
        

def main():

    training_data_file = open(r"C:\Users\ahasa\OneDrive\Asiakirjat\machine_learning\basic_network\mnist.csv")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    all_values = training_data_list[1].split(',')

    pixels = (np.asfarray(all_values[:784]) / 255 * 0.99) + 0.01
    pixels = pixels.reshape(28, 28) 

    #   double-padding to turn the input into a 32x32 matrix
    pixels = np.pad(pixels, (2, 2), 'constant')

    # plt.imshow(pixels)
    # plt.show()

    C1 = Conv_Layer(5, 6, 1)
    convo_1 = C1.forward_propagation(pixels)
    convo_1 = C1.activation_ReLu(convo_1)

    P1 = Pool_Layer("avg")
    pool_1 = P1.forward_propagation(convo_1)
    pool_1 = C1.activation_ReLu(pool_1)

    C2 = Conv_Layer(5, 16, 1)
    convo_2 = C2.forward_propagation(pool_1)
    convo_2 = C2.activation_ReLu(convo_2)

    P2 = Pool_Layer("avg")
    pool_2 = P2.forward_propagation(convo_2)
    pool_2 = C2.activation_ReLu(pool_2)

    C3 = Conv_Layer(5, 120, 1)
    convo_3 = C3.forward_propagation(pool_2)
    convo_3 = C3.activation_ReLu(convo_3)

    print(convo_3.shape)

if __name__ == '__main__':
    main()