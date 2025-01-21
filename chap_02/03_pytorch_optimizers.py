import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size: int):
        super(MLP, self).__init__() # lets the user specify the input size
        self.hidden_layer = nn.Linear(input_size, 64) # creates the neural net layers
        self.output_layer = nn.Linear(64, 2) # defines the size of the layer as well as the number of nodes the layer has. The two numbers determines the numbers of weights in the layer.
        self.activation = nn.ReLU() # defines an activation function

    def forward(self, x: int):
        """
        Defines how the components are actually connected to each others.
        """

        x = self.activation(self.hidden_layer(x))
        return self.output_layer(x)


model = MLP(input_size=10) # instantiate the multi-layer perceptron


# 1. Stochastic Gradient Descent Optimizer

# assuming `model` is your defined neural network
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# lr=0.01 sets the learning rate to 0.01
# momentum=0.9 smooths out updates and can help learning


# 2. Adam Optimizer

# assuming `model` is your defined neural network
optimizer = optim.Adam(model.parameters(), lr=0.001)

# lr=0.001 sets the learning rate to 0.001