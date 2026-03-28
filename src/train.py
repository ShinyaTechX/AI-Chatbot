# Simple pipeline: Text -> Tokenize -> Stem -> Bag of Words -> Neural Network -> Intent
# Bag of Words -> text -> numbers
# Neural Network -> learns patterns
# Loss -> how wrong model is
# Optimizer -> improves model
# Epoch -> one full training pass

# Import libraries
# torch -> build & train neural network
# json -> load intents file
# numpy -> work with arrays
# tokenize -> split sentence into words
# stem -> reduce words to root(e.g., running -> run)
# bag_of_words -> convert sentence -> numbers
# NeuralNet -> your neural network model

import torch
import json
import numpy as np

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

with open("../data/intents.json") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = [stem(w) for w in all_words if w.isalpha()]
all_words = sorted(set(all_words))      # Removes duplicates + sorts
tags = sorted(set(tags))

X_train = []
y_train = []

for pattern_sentence, tag in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

# Training
# Define model size
input_size = len(X_train[0])    # Number of features(vocabulary size)
hidden_size = 8     # Number of neurons in hidden layer
output_size = len(tags)     # Number of classes(intents)

model = NeuralNet(input_size, hidden_size, output_size)

criterion = torch.nn.CrossEntropyLoss()     # Measures error between prediction and correct answer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)      # Updates weights using Adam algorithm

for epoch in range(1000):   # Train model 1000 times
    inputs = torch.from_numpy(X_train)
    labels = torch.from_numpy(y_train)

    outputs = model(inputs)
    loss = criterion(outputs, labels)   # Compute loss: Calculates how wrong the model is
    # Backpropagation
    optimizer.zero_grad()   # reset old gradients   
    loss.backward()     # compute gradients
    optimizer.step()    # update weights

print("Training complete!")

# Save the model
torch.save(
    {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags,
    },
    "../model/chatbot_model.pth",
)
