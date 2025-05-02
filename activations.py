import numpy as np
from layer import Layer
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)
    
    def __str__(self):
        return super().__str__("Tanh")

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return  1/(1 + (2.7182818284590452353602874713527)**(-x))
        
        def sigmoid_prime(x):
            return sigmoid(x)*(1 - sigmoid(x))
        
        super().__init__(sigmoid, sigmoid_prime)
    
    def __str__(self):
        return super().__str__("Sigmoid")

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)
        
        def relu_prime(x):
            return np.where(x > 0, 1, 0)
            
        super().__init__(relu, relu_prime)

    def __str__(self):
        return super().__str__("ReLU")

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        def leaky_relu(x):
            return np.where(x > 0, x, x * alpha)
        
        def leaky_relu_prime(x):
            return np.where(x > 0, 1, alpha)
            
        self.alpha = alpha
        super().__init__(leaky_relu, leaky_relu_prime)

    def __str__(self):
        return super().__str__("LeakyReLU")
