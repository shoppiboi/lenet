import numpy as np
import layerlibrary as ll
import activationlibrary as ac
from matplotlib import pyplot as plt

def display_feature_maps(inputs):
    for x in range(inputs.shape[0]):
        plt.imshow(inputs[x])
        plt.show()

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
            "C1" : ll.Convolutional(5, 6, 32),
            "P1" : ll.Pooling("max"),
            "C2" : ll.Convolutional(5, 16, 28),
            "P2" : ll.Pooling("max"),
            "C3" : ll.Convolutional(5, 120, 14),
            "FC1" : ll.FullyConnectedLayer(120, 84),
            "FC2" : ll.FullyConnectedLayer(84, 10)
    }

    #   apply forward propagation for all layers in the dictionary
    out_cv1 = layers["C1"].forward_propagation(pixels)
    out_cv1 = ac.softmax(out_cv1)

    out_pool1 = layers["P1"].forward_propagation(out_cv1)
    out_pool1 = ac.softmax(out_pool1)

    out_cv2 = layers["C2"].forward_propagation(out_pool1) 
    out_cv2 = ac.softmax(out_cv2)

    out_pool2 = layers["P2"].forward_propagation(out_cv2)
    out_pool2 = ac.softmax(out_pool2)

    out_cv3 = layers["C3"].forward_propagation(out_pool2)
    out_cv3 = ac.softmax(out_cv3).flatten()

    out_cv3 = np.array(out_cv3, ndmin=2).T
    out_fc1 = layers["FC1"].forward_propagation(out_cv3)
    out_fc1 = ac.softmax(out_fc1).flatten()

    out_fc1 = np.array(out_fc1, ndmin=2).T
    out_fc2 = layers["FC2"].forward_propagation(out_fc1)

    final_output = ac.softmax(out_fc2)

    #   apply backpropagation for all layers in the dictionary
    output_errors = targets - final_output
    
    ll.softmax_derivative(np.array([1.8658, 2.2292, 2.8204]))

    err_fc1 = np.dot(layers["FC2"].weights.T, output_errors)

    layers["FC2"].back_propagation(output_errors, final_output, out_fc1)

    layers["FC1"].back_propagation(err_fc1, out_fc1, out_cv3)

if __name__ == '__main__':
    main()