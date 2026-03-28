# Import PyTorch. torch -> main PyTorch library. torch.nn -> contains neural network layers and functions.
# Create a class called NeuralNet. Inherits from nn.Module -> base class for all PyTorch models. Think: Every PyTorch model is a Module. 

# input_size -> number of input features(length of bag-of-words vector)
# hidden_size -> number of neurons in hidden layers
# num_classes -> number of output classes(number of intents)
# super() -> initializes the parent nn.Module
# l1: input -> Hidden layer1
# l2: Hidden layer1 -> Hidden layer2
# l3: Hidden layer2 -> Output layer(num_classes neurons)
# ReLU: Activation function -> adds non-linearity

# forward(self, x): Forward pass
# Defines how data flows through the network. Must return output predictions
# out = self.l1(x): input -> first layer. Linear transformation: input_size -> hidden_size
# out = self.relu(out): ReLU activation. Applies non-linearity
# out = self.l2(out): Second layer. hidden_size -> hidden_size
# out = self.relu(out): ReLU again
# out = self.l3(out): Output layer. hidden_size -> num_classes. Each neuron corresponds to a class(intent)
# return out: Return output. Raw scores(logits) for each class. Later we can apply Softmax to get probabilities

# Diagram of flow 
# Input(bag-of-words vector) - l1(Linear) - ReLU - l2(Linear) - ReLU - l3(Linear) - Output logits(num_classes)

import torch
import torch.nn as nn
 
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out