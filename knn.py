import sys
import numpy as np

# Reference
# https://stackoverflow.com/questions/2545397/converting-a-string-into-a-list-in-python

k = 5

# This function extracts the data from input file (train/test) and returns 3 vectors (names,orientation,rgb values)
def get_vectors(file):
    file = open(file, "r")
    ids = []
    orientation = []
    rgb_vectors = []

    for i in file:
        temp_list = i.split()
        ids.append(temp_list[0])
        orientation.append(temp_list[1])
        rgb_vectors.append(np.array(temp_list[2:]).astype(int))
        # print(ids)
        # print(orientation)
        # print(rgb_vectors[0])
    return ids, orientation, rgb_vectors


# Train function ( copy data from train file to model file )
def train_model(input_file, output_file):
    with open(input_file) as input:
        with open(output_file, "w") as out:
            for line in input:
                out.write(line)


#  Testing model for KNN
def test_model(input_file, model_file):
    train_ids, train_orientations, train_rgb_vectors = get_vectors(model_file)
    test_ids, test_orientations, test_rgb_vectors = get_vectors(input_file)
    true_count = 0
    result = []
    file = open("output_nearest.txt", 'w')

    # For each test-image find Euclidean distance to train images and pick k small values
    for i in range(len(test_ids)):
        dist = []
        for j in range(len(train_ids)):
            euclidean_dis = np.linalg.norm(test_rgb_vectors[i] - train_rgb_vectors[j])
            dist.append(euclidean_dis)
        sorted_list = dist.copy()
        sorted_list.sort()

        k_list = sorted_list[:k]
        orientation_k = []
        for x in k_list:
            index = dist.index(x)
            orientation_k.append(train_orientations[index])
        predicted_orientation = max(set(orientation_k), key=orientation_k.count)
        # sys.exit(0)
        if predicted_orientation == test_orientations[i]:
            true_count += 1;
        # print(true_count)
        file.write(str(test_ids[i]))
        file.write(' ')
        file.write(predicted_orientation + " \n")

    file.close()
    accuracy_nearest = (true_count / len(test_ids)) * 100
    print("Accuracy for KNN is : ", accuracy_nearest)
