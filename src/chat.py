import torch
import random
import json

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

data = torch.load("../model/chatbot_model.pth")

model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

all_words = data["all_words"]
tags = data["tags"]

with open("../data/intents.json") as f:
    intents = json.load(f)

print("Chatbot ready! (type 'quit')")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    X = bag_of_words(tokenize(sentence), all_words)
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=0)

    tag = tags[predicted.item()]
    
    for intent in intents["intents"]:
        if tag == intent["tag"]:
            print("Bot:", random.choice(intent["responses"]))
