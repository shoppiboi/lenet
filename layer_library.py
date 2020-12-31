import numpy as np
from matplotlib import pyplot as plt

class Conv_Layer:
    def __init__(self, filter_size=0, filter_count=0, stride=0):
        self.size = filter_size
        self.filter_count = filter_count
        self.stride = stride
        self.filters = np.random.normal(0.0, pow(self.size, -0.5), (self.filter_count, self.size, self.size))

    def forward_propagation(self, inputs):
        
        #   calculate the size of the output matrix
        #   based on the formula: (x-axis - filter_size) / stride + 1
        resulting_dimension = int(  (inputs.shape[0]-self.size)/self.stride+1   )

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


        for x in range(6):
            resulting_matrix[x][resulting_matrix[x]<=0] = 0
            plt.imshow(resulting_matrix[x])
            plt.show()

        return resulting_matrix

    def backward_propagation(self):
        pass

    def activation_ReLu(self, inputs):
        for x in range(inputs.shape[0]):
            inputs[x][inputs[x]<=0]=0

        return inputs

class Pool_Layer:
    def __init__(self):
        pass

    def forward_propagation(self):
        pass


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

    layer = Conv_Layer(5, 6, 1)
    convo1 = layer.forward_propagation(pixels)


if __name__ == '__main__':
    main()