#!/usr/bin/python3
import numpy as np
import json
from layer import Layer
from matplotlib import pyplot as plt 

LEARNING_RATE = 0.05
MAX_EPOCH = 100

layers = []

def load_data(fname):
    x = []
    y = []

    train_data_file = open(fname, 'r')
    train_data = json.load(train_data_file)
    for data_piece in train_data:
        data = []
        for data_number in data_piece['data']:
            data.append(int(data_number))
        data_mat = np.array(data)
        data_mat = data_mat.T
        x.append(data_mat)

        y_data = [0, 1] if data_piece['ans'] == '1' else [1, 0]
        y.append(np.array(y_data))
    return x, y

def init_layers():
    global layers
    layers = [
        Layer(3, 12),
        Layer(2, 3, True),
    ]
    pass

def cost(predict, y):
    if predict[0][0] > predict[1][0]:
        predict[0][0] = 1
        predict[1][0] = 0
    else:
        predict[0][0] = 0
        predict[1][0] = 1
    y = y.reshape(2, 1)
    return np.sum((y - predict) ** 2) / 2

def feed(x, y):
    epoches = []
    costs = []

    for epoch in range(MAX_EPOCH):
        epoch_cost = 0
        for (x_, y_) in zip(x, y):
            output = x_.reshape(12, 1)
            deltas = y_.reshape(2, 1)
            # feed every layers (DO NOT WRITE LIKE THIS)
            for layer in layers:
                output = layer.feed(output)
            # back propagation of deltas
            for layer in layers[::-1]:
                deltas = layer.layer_error(deltas)
            epoch_cost += cost(output, y_)
        # batch gradient descent for every layers
        for layer in layers[::-1]:
            layer.correction(LEARNING_RATE)

        epoches.append(epoch)
        costs.append(epoch_cost)

        if epoch_cost == 0:
            break

    plt.xlabel("Epoches")
    plt.ylabel("Costs")
    plt.plot(epoches, costs)
    plt.show()

def test(x, y):
    for (test_case_x, test_case_y) in zip(x, y):
        inputs = test_case_x.reshape(12, 1)
        delta = test_case_y.reshape(2, 1)
        for layer in layers:
            inputs = layer.test(inputs)
        ans = 0 if delta[0][0] == 1 else 1
        
        print('  NN gives %d, and the answer is %d.' % (inputs, ans))

if __name__ == '__main__':
    x, y = load_data('./data/train.json')
    init_layers()

    print('Training...')
    feed(x, y)

    x_test, y_test = load_data('./data/test.json')
    print('Testing...')
    test(x_test, y_test)

# print('train finished, test case: {}, and predit: {}.', y[0], test(x[0]))