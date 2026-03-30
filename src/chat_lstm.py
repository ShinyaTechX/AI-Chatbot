# Full Flow(Simple)
# User: "Hello"
# → Tokenize → ["Hello"]
# → Encode → [1]
# → Pad → [1, 0, 0, ..., 0]
# → Tensor → (1, 10, 1)
# → LSTM → prediction
# → Tag → "greeting"
# → Response → "Hi there!"

import torch
import json
import random

from nltk_utils import tokenize, stem
from model_lstm import LSTMModel

data = torch.load("../model/lstm_chatbot.pth")   # Load trained model. This loads everything you saved earlier: {"model": weights, "word2idx": dictionary, "tags": list_of_tags}

# Extract saved data
word2idx = data["word2idx"]   # word2idx -> converts words -> numbers
tags = data["tags"]   # tags -> maps prediction index -> intent name

model = LSTMModel(input_size=1, hidden_size=8, output_size=len(tags))   # Create same model structure as training
model.load_state_dict(data["model"])   # Load learned weights
model.eval()   # switch to inference mode(no training)

with open("../data/intents.json") as f:
    intents = json.load(f)

def encode(sentence):   # Converts words -> numbers. Unknow words -> 0
    return [word2idx.get(stem(w), 0) for w in sentence]

def pad(seq, max_len=10):   # Ensures input length = 10
    return seq + [0]*(max_len - len(seq))   # Same as training

print("LSTM Chatbot Ready")

while True:   # Chatbot loop
    sentence = input("You: ")
    if sentence == "quit":
        break

    # Process user input. Input: "Hello there"
    tokens = tokenize(sentence)   # ["Hello", "there"]
    encoded = encode(tokens)   # Encoded -> [1, 0]
    padded = pad(encoded)   # Padded -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    X = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(2)   # Convert to tensor. Step by step shape: 1. Before: [1, 0, 0, ..., 0] -> (10,)   2. unsqueeze(0) -> add batch dimension: (1, 10)   3. unsqueeze(2) -> add input_size: (1, 10, 1)   Final shape matches LSTM input: (batch_size, seq_len, input_size)

    # Model prediction
    output = model(X)   # output -> logits(scores for each tag)
    _, predicted = torch.max(output, dim=1)   # torch.max -> picks highest score. Example: output = [2.1, 0.3, 1.7], predicted = 0

    tag = tags[predicted.item()]   # Get predicted tag. Convert index -> actual label. Example: predicted = 0, tags = ["greeting", "bye", "thanks"] -> tag = "greeting"

    for intent in intents["intents"]:   # Select response
        if intent["tag"] == tag:   # Find matching intent
            print("Bot:", random.choice(intent["responses"]))   # Choose random response. Example: "greeting" -> ["Hi!", "Hello!", "Hey!"]. Bot: "Hello!"