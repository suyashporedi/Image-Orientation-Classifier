import pandas as pd
import numpy as np
import random
import sys


def classify_dt(subnode1, subnode2, c, v, d):
    d = d + 1
    if len(subnode1) == 0:
        return np.unique(c)[np.argmax(np.unique(c, return_counts=True)[1])]

    elif d == 5:
        return np.unique(subnode1)[np.argmax(np.unique(subnode1, return_counts=True)[1])]

    else:
        best_info_gain = 0
        for i in range(len(subnode2[0])):
            gain = subnode2[:, i]
            split_col_value = subnode1
            print(gain,best_info_gain)
            if gain > best_info_gain:
                best_info_gain = gain
                split_value1 = split_col_value
                col_ind = i
            else:
                continue
        best_col = col_ind
        tree = {best_col: {}}
        data = np.column_stack((subnode1, subnode2))
        X = subnode2
        Y = subnode1
        vi = [data[data[:, best_col] <= split_value1, :], data[data[:, best_col] > split_value1, :]]
        for j in range(len(vi)):
            part = vi[j]
            sub_x = part[:, 1:]
            sub_y = part[:, 0]
            sub_tree = classify_dt(sub_y, sub_x, Y, X, d)
            tree[best_col][str(split_value1) + '_' + str(j)] = sub_tree
        return (tree)


def get_value(input):
    value = input.iloc[:, 1:]
    key = input.iloc[:, 0]
    return np.array(key), np.array(value)


def find_v(l, tree, default=0):
    for key in list(l.keys()):
        if key in tree.keys():
            try:
                z = list(tree[key])[0]
                n = int(z.rsplit('_', 1)[0])
                if l[key] > n:
                    v = tree[key][str(n) + '_' + str(1)]
                    # print(result)
                else:
                    v = tree[key][str(n) + '_' + str(0)]
            except:
                return 0

            res = v
            if isinstance(res, dict):

                return find_v(l, v)
            else:
                return res


def test_result(input, dt):
    data = input.iloc[:, 2:]
    new_col = [i for i in range(len(data.columns))]
    data.columns = new_col
    q = data.to_dict(orient="records")
    predicted = pd.DataFrame()
    for i in range(len(data)):
        predicted.loc[i, ""] = find_v(q[i], dt, 1.0)
    print('Accuracy for Decision tree is ', (np.sum(predicted[""] == input.iloc[:, 1]) / len(input)) * 100, '%')
    return predicted


if __name__ == "__main__":

    if sys.argv[4] == 'tree':
        if sys.argv[1] == 'train':
            train_file = str(sys.argv[2])
            train_input = pd.read_table(train_file, sep=" ", header=None)
            key, val = get_value(train_input)
            tree = classify_dt(val[:, 0], val[:, 1:], val[:, 0], val[:, 1:], 0)
            file = open(sys.argv[3], "w")
            file.close()

        if sys.argv[1] == 'test':
            test_file = str(sys.argv[2])
            test_input = pd.read_table(test_file, sep=" ", header=None)
            file = open(sys.argv[3], "r")
            dt = eval(file.read())
            file.close()
            key1 = test_input.iloc[:, 0]

            orientation = test_result(test_input, dt)

            dt_file = pd.concat([key1, orientation], axis=1)

            output_file = open("output_dt.txt", 'a')
            output_file.write(dt_file.to_string())
            output_file.close()
