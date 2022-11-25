import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward_propagation(self, input):
        pass

    def backward_propagation(self, output_error, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5   #number of input and output neurons
        self.bias = np.random.rand(1, output_size) - 0.5
        
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T) 
        weights_error = np.dot(self.input.T, output_error)  
        self.weights -= learning_rate * weights_error       
        self.bias -= learning_rate * output_error
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation_func, activation_func_prime):
        self.activation = activation_func
        self.activation_prime = activation_func_prime

    def forward_propagation(self, data):
        self.input = data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return 1. * (x > 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def softmax_prime(x):
    return softmax(x) * (1 - softmax(x))

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        len_sam = len(input_data)
        result = []
        for i in range(len_sam):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, learning_rate, x_val, y_val):
        len_sam = len(x_train)
        error_train = []
        error_val = []
        # training loop
        for i in range(epochs):
            train_err = 0
            for j in range(len_sam):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                train_err += self.loss(y_train[j], output)
                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            train_err /= len_sam
            error_train.append(train_err)
            validation_err = self.loss(y_val, self.predict(x_val))
            error_val.append(validation_err)
            print('epoch: %d/%d   t_error: %f val_error: %f' % (i+1, epochs, train_err, validation_err))
        return error_train, error_val  