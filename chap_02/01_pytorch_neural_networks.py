import torch
import torch.nn as nn


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

# print(model)
# MLP(
#   (hidden_layer): Linear(in_features=10, out_features=64, bias=True)
#   (output_layer): Linear(in_features=64, out_features=2, bias=True)
#   (activation): ReLU()
# )

print(model.forward(torch.rand(10)))
# tensor([0.2294, 0.2650], grad_fn=<AddBackward0>)
