#!/usr/bin/python3
import numpy as np
import json
from layer import Layer
from matplotlib import pyplot as plt 

x, y = [], []
layers = []

def load_data():
    train_data_file = open('./data/train.json', 'r')
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

def init_layers():
    global layers
    layers = [
        Layer(3, 12),
        Layer(2, 3, True),
    ]
    pass

def cost(predict, y):
    return np.sum((y - predict) ** 2) / 2

def feed(x, y):
    epoches = []
    costs = []

    for epoch in range(1000):
        epoch_cost = 0
        for (x_, y_) in zip(x, y):
            output = x_.reshape(12, 1)
            deltas = y_.reshape(2, 1)
            for layer in layers:
                output = layer.feed(output)
            for layer in layers[::-1]:
                deltas = layer.layer_error(deltas, 0.1)
            epoch_cost += cost(output, y_)
        for layer in layers[::-1]:
            layer.correction(64)

        epoches.append(epoch)
        costs.append(epoch_cost)

    plt.xlabel("Epoches")
    plt.ylabel("Costs")
    plt.plot(epoches, costs)
    plt.show()
    

def test(x):
    for layer in layers:
        x = layer.give_output(x)
    return x


if __name__ == '__main__':
    load_data()
    init_layers()
    feed(x, y)

# print('train finished, test case: {}, and predit: {}.', y[0], test(x[0]))