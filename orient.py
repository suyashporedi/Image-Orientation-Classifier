#!/usr/bin/env python3
###################################
# CS B551 Fall 2019, Assignment #4
#
# Authors :
# Suyash Poredi - sporedi,
# Deepali Tolia - dtolia,
# Kaustubh Bhalerao - kbhaler
#

import sys
import knn, nnet
import time
import numpy as np
import pandas as pd
import random

# Reference - For Decision tree - Discussed Logic with Shreyas Bhujbal's Team

def gini_index(y):
    unique, counts = np.unique(y, return_counts=True)
    sum_sqr = np.sum([(counts[i] / sum(counts)) ** 2 for i in range(len(unique))])
    gini = 1 - sum_sqr
    return gini


def calculate_info_gain(x, y):
    total_gini = gini_index(y)
    unique_y, counts_y = np.unique(y, return_counts=True)
    random_x = random.sample(range(min(x), max(x)), 5)
    data = np.column_stack((x, y))
    best_gain = 0
    a = x
    b = y
    for k in random_x:
        q_l = len(a[a <= k]) / len(a)
        q_r = len(a[a > k]) / len(a)
        subset_l = data[data[:, 0] <= k, :]
        subset_r = data[data[:, 0] > k, :]
        x_l = subset_l[:, 0]
        y_l = subset_l[:, 1]
        x_r = subset_r[:, 0]
        y_r = subset_r[:, 1]
        unique_l, counts_l = np.unique(y_l, return_counts=True)
        gini_l = 1 - np.sum([(counts_l[i] / sum(counts_l)) ** 2 for i in range(len(unique_l))])
        unique_r, counts_r = np.unique(y_r, return_counts=True)
        gini_r = 1 - np.sum([(counts_r[j] / sum(counts_r)) ** 2 for j in range(len(unique_r))])
        info_gain = total_gini - (q_l * gini_l + q_r * gini_r)
        if info_gain > best_gain:
            best_gain = info_gain
            split_value = k
        else:
            continue
    return best_gain, split_value


def Decision_tree_classifier(y_sub, x_sub, y, x, d):
    d = d + 1
    if len(y_sub) == 0:
        return np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]

    elif d == 5:
        return np.unique(y_sub)[np.argmax(np.unique(y_sub, return_counts=True)[1])]

    else:
        best_info_gain = 0
        for i in range(len(x_sub[0])):
            gain, split_col_value = calculate_info_gain(x_sub[:, i], y_sub)
            if gain > best_info_gain:
                best_info_gain = gain
                split_value1 = split_col_value
                col_ind = i
            else:
                continue
        best_col = col_ind
        tree = {best_col: {}}
        data = np.column_stack((y_sub, x_sub))
        X = x_sub
        Y = y_sub
        a = [data[data[:, best_col] <= split_value1, :], data[data[:, best_col] > split_value1, :]]
        for j in range(len(a)):
            part = a[j]
            sub_x = part[:, 1:]
            sub_y = part[:, 0]
            sub_tree = Decision_tree_classifier(sub_y, sub_x, Y, X, d)
            tree[best_col][str(split_value1) + '_' + str(j)] = sub_tree
        return (tree)


def predict(query, tree, default=0):
    for key in list(query.keys()):
        if key in tree.keys():
            try:
                z = list(tree[key])[0]
                n = int(z.rsplit('_', 1)[0])
                if query[key] > n:
                    result = tree[key][str(n) + '_' + str(1)]
                else:
                    result = tree[key][str(n) + '_' + str(0)]
            except:
                return 0

            result1 = result
            if isinstance(result1, dict):
                return predict(query, result)
            else:
                return result1


def test_predict(data, tree):
    data1 = data.iloc[:, 2:]
    new_col = [i for i in range(len(data1.columns))]
    data1.columns = new_col
    queries = data1.to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])
    for i in range(len(data1)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 1.0)
    print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data.iloc[:, 1]) / len(data)) * 100, '%')
    return predicted


def get_value(input):
    value = input.iloc[:, 1:]
    key = input.iloc[:, 0]
    return np.array(key), np.array(value)


if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) < 4:
        print("The program expects 4 arguments eg : orient.py test test-data.txt nnet_model.npz nnet")
        sys.exit()

    test_train_option = str(sys.argv[1])
    input_file = str(sys.argv[2])
    model_file_path = str(sys.argv[3])
    model_approach = str(sys.argv[4])



    if model_approach == "nearest":
        if test_train_option == "train":
            knn.train_model(input_file, model_file_path)
        elif test_train_option == "test":
            knn.test_model(input_file, model_file_path)

    if sys.argv[4] == 'tree':
        if sys.argv[1] == 'train':
            data = pd.read_table(sys.argv[2], sep=" ", header=None)
            key, data1 = get_value(data)
            tree = Decision_tree_classifier(data1[:, 0], data1[:, 1:], data1[:, 0], data1[:, 1:], 0)
            f = open(sys.argv[3], "w")
            print(tree, file=f)
            f.close()

        if sys.argv[1] == 'test':
            test = pd.read_table(sys.argv[2], sep=" ", header=None)
            f = open(sys.argv[3], "r")
            tree1 = eval(f.read())
            f.close()
            key1 = test.iloc[:, 0]
            actual = test.iloc[:, 1]
            pred = test_predict(test, tree1)

            pred_df = pd.concat([key1, actual, pred], axis=1)

            file = open("output_tree.txt", 'a')
            file.write(pred_df.to_string())
            file.close()

    if model_approach == "nnet":
        if test_train_option == "train":
            nnet.train_model(input_file, model_file_path)
        elif test_train_option == "test":
            nnet.test_model(input_file, model_file_path)

    if model_approach == "best":
        if test_train_option == "train":
            nnet.train_model(input_file, model_file_path)
        elif test_train_option == "test":
            nnet.test_model(input_file, model_file_path)

    print("\ntime required : ", time.time() - start_time)
