import copy

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def __str__(self):
        pass

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

    