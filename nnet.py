


import numpy as np
import sys

orientation_dict = {0: 0, 90: 1, 180: 2, 270: 3}
hidden_nodes = 100
iterations = 20

# training the model with stocastic graident descent approach.
# Whole data is considered as size of data is not much i.e approx 360000rows
def train_model(input_file, output_file):
    image_names, orientations_array, rgb_vector_array = get_array(input_file)

    #  Resetting the random seeds to generate unique values each-time
    np.random.seed(1)

    #  Generating Random values for Weight and Bias

    w01 = np.random.random((len(rgb_vector_array[0]), hidden_nodes))
    w12 = np.random.random((hidden_nodes, len(orientations_array[0])))

    b01 = np.random.random((1, hidden_nodes))
    b12 = np.random.random((1, len(orientations_array[0])))

    learning_rate = 0.045

    for iter in range(iterations):
        for i in range(len(rgb_vector_array)):
            A = np.array(rgb_vector_array[i: min((i + 1), len(rgb_vector_array)), :])
            B = np.array(orientations_array[i: min((i + 1), len(rgb_vector_array)), :])

            # Feed Forward and Layer Formation (3 Layers) 1- Input  2- Hidden 3- Output
            input_layer = A
            hidden_layer = (np.dot(input_layer, w01) + b01)
            output_layer = (np.dot(sigmoid_function(hidden_layer), w12) + b12)

            # Backpropagation

            output_back = (B - sigmoid_function(output_layer)) * sigmoid_function(output_layer)
            b12 += learning_rate ** 1 * output_back

            hidden_back = output_back.dot(w12.T) * sigmoid_function(hidden_layer)
            b01 += learning_rate ** 1 * hidden_back

            # Updating the weight values
            w12 += learning_rate * sigmoid_function(hidden_layer).T.dot(output_back)
            w01 += learning_rate * input_layer.T.dot(hidden_back)  # L0 -> L1

    np.savez_compressed(output_file, w01, w12, b01, b12)


#  Activation Function
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def tanh_function(x):
    return np.tanh(x)


def relu_function(x):
    return np.maximum(0, x)

def softmax_function(x):
    expo = np.exp(x)
    expo_sum = np.sum(np.exp(x))
    return expo / expo_sum

# Get array vectors from Input File
def get_array(path):
    file = open(path, 'r')
    names = []
    orientations = []
    rgb_data = []
    max_value = -1000
    for i in file:
        orientation_matrix = [0.0] * 4
        temp_list = i.split()
        vector = np.array(temp_list[2:]).astype(int)
        names.append(temp_list[0])
        if max_value < max(vector):
            max_value = max(vector)
        # sys.exit(0)
        rgb_data.append(vector)
        orientation_matrix[orientation_dict[int(temp_list[1])]] = 1
        orientations.append(orientation_matrix)

    return np.array(names), np.array(orientations), np.array(rgb_data) / max_value


def test_model(input_file, model_file):

    # Load file with the 4 Numpy Arrays
    load_data = np.load(model_file)

    w01 = load_data['arr_0']
    w12 = load_data['arr_1']
    b01 = load_data['arr_2']
    b12 = load_data['arr_3']

    image_names, orientations_array, rgb_vector_array = get_array(input_file)
    output_file = open('output_nnet.txt', 'w')
    correct_predicted = 0
    for i in range(len(rgb_vector_array)):
        layer0 = rgb_vector_array[i]
        layer1 = sigmoid_function(np.dot(layer0, w01) + b01)
        layer2 = sigmoid_function(np.dot(layer1, w12) + b12)
        prediction = layer2[0]
        print(prediction)
        # take the maximum value of orientation classified
        predicted_orientation = max([(prediction[x], x) for x in range(len(prediction))], key=lambda x: x[0])[1]
        actual_orientation = max([(orientations_array[i][x], x) for x in range(len(orientations_array[i]))],
                                 key=lambda x: x[0])[1]

        if predicted_orientation == actual_orientation:
            correct_predicted += 1

        output_file.write(image_names[i] + " " + str(predicted_orientation * 90) + "\n")

    output_file.close()

    print("Accuracy: ", ((correct_predicted / len(rgb_vector_array)) * 100))
