#!/usr/bin/python3
import math
import numpy as np

class Layer:
    neurons = 0
    activation = None
    d_activation = None

    # this layer to next layer
    weight = None

    # this layer
    bias = None

    is_output = False

    last_input_arr = None
    last_output_arr = None

    delta_bias = None
    delta_weight = None

    previous_layer_neurons = 0

    def __init__(self, neurons, previous_layer_neurons, is_output = False):
        self.activation = self.sigmoid
        self.d_activation = self.d_sigmoid
        self.is_output = is_output
        self.previous_layer_neurons = previous_layer_neurons

        self.neurons = neurons
        self.weight = np.random.random(size=(neurons, previous_layer_neurons))
        self.bias = np.random.random(size=(neurons, 1))

        self.delta_weight = np.zeros((neurons, previous_layer_neurons), dtype=float)
        self.delta_bias = np.zeros((neurons, 1), dtype=float)

    # returns error of previous layer
    def previous_layer_error(self, delta):
        return self.weight.T @ delta * self.d_activation(self.last_input_arr)

    # batch error backward propagation
    def layer_error(self, delta_or_t, learning_rate):
        if self.is_output:
            delta_or_t = self.last_output_arr - delta_or_t
        else:
            self.delta_bias += learning_rate * delta_or_t
            self.delta_weight += learning_rate * delta_or_t @ self.last_input_arr.T
        return self.previous_layer_error(delta_or_t)

    # batch gradient descent
    def correction(self, m):
        self.bias -= self.delta_bias / m
        self.weight -= self.delta_weight / m

        self.delta_weight = np.zeros((self.neurons, self.previous_layer_neurons), dtype=float)
        self.delta_bias = np.zeros((self.neurons, 1), dtype=float)

    # forward propagation
    def feed(self, a):
        # input of this layer
        z = self.weight @ a
        # activation
        r = self.activation(z + self.bias)
        self.last_input_arr = a
        self.last_output_arr = r
        return r

    @classmethod
    def sigmoid(cls, x):
        return 1 / (1 + np.exp(-x))
    
    @classmethod
    def d_sigmoid(cls, z):
        return cls.sigmoid(z) * (1 - cls.sigmoid(z))