# Import libraries
# torch -> load and run the model
# random -> pick random response
# json -> load intents
# NeuralNEt -> your trained model structure
# tokenize -> split sentence into words
# bag_of_words -> convert sentence -> vector

# Full pipline(simple)
# User input -> Tokenize -> Bag of Words -> Neural Network -> Predicted tag -> Find responses -> Print random reply 
# Example run: You: hello -> ["hello"] -> [1,0,0,...] -> model -> [2.3, 0.5] -> tag = "greeting" -> Bot: "Hi there!"

import torch
import random
import json

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

data = torch.load("../model/chatbot_model.pth")   # Load trained model. Loads saved file from training step

model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])   # Recreate model. Creates same neural network structure
model.load_state_dict(data["model_state"])   # Load learned weights. Loads trained weights & biases into model
model.eval()   # Set evaluation mode. disables training behavior. no dropout/randomness. stable predictions

all_words = data["all_words"]   # all_words -> vocabulary 
tags = data["tags"]   # tags -> intent labels 

with open("../data/intents.json") as f:
    intents = json.load(f)   # Load intents file. 

print("Chatbot ready! (type 'quit')")

while True:   # Runs forever until user exits
    sentence = input("You: ")   # Get user input
    if sentence == "quit":
        break

    X = bag_of_words(tokenize(sentence), all_words)   # Convert text -> numbers. Step-by-step: "hello" -> ["hello"] -> [1, 0, 0, ...]
    X = torch.from_numpy(X)   # Convert Numpy -> PyTorch tensor

    output = model(X)   # Neural network processes input. Output example: [2.1, 0.3, -1.2]: These are logits(scores)
    _, predicted = torch.max(output, dim=0)   # Get predicted class. Finds index of highest value. Example: [2.1, 0.3, -1.2] -> index 0. _ -> convention for ignoring the first value(we don't care abount actual score here). predicted -> stores the index of the predicted class

    tag = tags[predicted.item()]   # Convert index -> tag. Example: 0 -> "greeting"
    
    for intent in intents["intents"]:   # Find matching intent. Find correct intent in JSON
        if tag == intent["tag"]:
            print("Bot:", random.choice(intent["responses"]))   # Print random response. Example: "Hello!", "Hi there!"
