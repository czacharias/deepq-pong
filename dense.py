import numpy as np
from layer import Layer
import copy

class Dense(Layer):

    def __init__(self, input_size, output_size, weights=None, bias=None):
        self.input_size = input_size
        self.output_size = output_size
        limit = np.sqrt(6 / (input_size + output_size))
        
        self.weights = (np.random.uniform(-limit, limit, (output_size, input_size)) if weights is None else weights)
        self.bias = (np.zeros((output_size, 1)) if bias is None else bias)

    def __deepcopy__(self, memo):
        copied = Dense(self.input_size, self.output_size)
        copied.weights = copy.deepcopy(self.weights, memo)
        copied.bias = copy.deepcopy(self.bias, memo)
        return copied
    
    def __str__(self):
        return "Dense " + self.input_size + "x" + self.output_size

    def forward(self, input):
        input_array = np.array(input, dtype=float)
        
        if input_array.ndim == 1:
            self.input = input_array.reshape(-1, 1)
        else:
            self.input = input_array
        
        output = np.dot(self.weights, self.input) + self.bias
        return output.flatten()
    
    def backward(self, output_gradient, learning_rate):
        if output_gradient.ndim == 1:
            output_gradient = output_gradient.reshape(-1, 1)
            
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        
        return input_gradient.flatten()
