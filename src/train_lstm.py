# Summary of the flow
# 1. Load intents JSON
# 2. Tokenize and stem patterns
# 3. Build vocabulary(all_words) and tags
# 4. Encode patterns as sequences of integers
# 5. Pad sequences to same length
# 6. Reshape for LSTM input
# 7. Train LSTM with CrossEntropyLoss
# 8. Save model and data for inference

# visualize how your sentence flows the LSTM chatbot step by step.
## Step1: Input sentence 
# Example: "Hello how are you"
# Tokenize -> ["Hello", "how", "are", "you"]
# Stem -> ["hello", "how", "are", "you"] (after stemming, some words may change)
# Encode -> [1, 5, 3, 8] (numbers from word2idx)
# Pad to max_len=10 -> [1, 5, 3, 8, 0, 0, 0, 0, 0, 0]
# Reshape for LSTM -> (seq_len=10, input_size=1)
## Step2: Input shape to LSTM
# LSTM expects shape (batch_size, seq_len, input_size)
# Here batch_size=1(one sentence)
# seq_len=10, input_size=1
# Input Tensor: [[[1], [5], [3], [8], [0], [0], [0], [0], [0], [0]]], Shape: (1, 10, 1)
## Step3: LSTM layer
# Processes the sequence one word at a time(time steps)
# Hidden size=8 -> LSTM outputs 8 features per time step
# Time Step 1: word=1 → LSTM outputs h1 (8 features)
# Time Step 2: word=5 → LSTM outputs h2 (8 features)
# ...
# Time Step 10: word=0 → LSTM outputs h10 (8 features)
# LSTM also maintains a cell state to remember context across words
## Step4: Take last hidden state
# Usually, the last hidden state(h10) is passed to the next layer
# Shape: (1, hidden_size = 8)
## Step5: Fully Connected(Linear) layer
# Maps 8 hidden features -> number of classes(tags)
# Linear Layer:
# Input: 8 features
# Output: len(tags)=5 (for example)
# Output Example: [2.1, 0.5, 0.2, 1.8, 0.3]
# These are logits for each intent
## Step6: Prediction
# Apply softmax to convert logits -> probabilities: [0.45, 0.10, 0.05, 0.35, 0.05]
# Highest probability -> predicted intent: "greeting"
## Step7: Summary Diagram
# Input Sentence: "Hello how are you"
#         |
#       Tokenize
#         |
#       Stem
#         |
#     Encode → [1, 5, 3, 8]
#         |
#       Pad → [1, 5, 3, 8, 0, 0, 0, 0, 0, 0]
#         |
#   Reshape → (1, 10, 1)
#         |
#       LSTM Layer (hidden_size=8)
#         |
#     Last Hidden State (1, 8)
#         |
#   Fully Connected Layer → logits (1, num_tags)
#         |
#      Softmax → probabilities
#         |
#   Predicted Tag → "greeting"

# Key ida: Each word is converted to a number → fed sequentially into the LSTM → LSTM remembers context → outputs are classified into intents.

import torch
import json
import numpy as np

from nltk_utils import tokenize, stem
from model_lstm import LSTMModel

# Load data
with open("../data/intents.json") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stemming and sorting words
all_words = sorted(set([stem(w) for w in all_words if w.isalpha()]))   
tags = sorted(set(tags))   # w.isalpha() -> ignores punctuation/numbers. stem(w) -> reduces each word to its root form. set() -> removes duplicates. sorted() -> keeps consistent order for reproducibility. Example: ["hello", "hello", "running"] -> ["hello", "run"]

# Create a word index
word2idx = {w: i+1 for i, w in enumerate(all_words)}   # Assigns each word a unique index. i+1 -> Starts indexing from 1(0 is reserved for padding). Example: all_words = ["hello", "run", "bye"]. word2idx -> {"hello":1, "run":2, "bye":3}

def encode(sentence):   # Encode sentences as sequences of integers
    return [word2idx.get(stem(w), 0) for w in sentence]   # Converts a tokenized sentence to a sequence of integers based on word2idx. 0 is used for unknown words(not in word2idx). Example: sentence = ["Hello", "there"], encode(sentence) -> [1, 0]: assuming "there" is not in word2idx

X = []
y = []

for (pattern, tag) in xy:
    encoded = encode(pattern)
    X.append(encoded)
    y.append(tags.index(tag))

# Padding
# LSTM requires sequences of same length, so we pad shorter sequences with 0
# max_len = 10 -> sequences longer than 10 would need truncation(not done here)
# Example: [1, 2] -> pad -> [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
max_len = 10

def pad(seq):
    return seq + [0]*(max_len - len(seq))

X = np.array([pad(x) for x in X])
y = np.array(y)

# Reshape for LSTM
X = np.expand_dims(X, axis=2)   # (batch, seq_len, 1). LSTM expects input shape: (batch_size, sequence_length, input_size). input_size = 1 because each word is represented by a single integer

X = torch.tensor(X, dtype=torch.float32)   # Then convert to PyTorch tensors
y = torch.tensor(y, dtype=torch.long)

# Model 
model = LSTMModel(input_size=1, hidden_size=8, output_size=len(tags))   # Define the model. input_size=1 -> each word is a single integer. hidden_size=8 -> number of hidden units in LSTM. output_size=len(tags) -> number of possible intents to classify

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()   # CrossEntropyLoss -> suitable for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # Adam -> optimizer for training the network

# Train
for epoch in range(500):   # Training loop. Repeats for 500 epochs
    outputs = model(X)   # forward pass: model predicts logits for each tag
    loss = criterion(outputs, y)   # compute loss

    optimizer.zero_grad()   # clear previous gradients
    loss.backward()   # compute gradients via backpropagation
    optimizer.step()   # update model weights

print("LSTM Training Done!")

torch.save({   # Save the trained model
    "model": model.state_dict(),   # trained LSTM weights
    "word2idx": word2idx,   # for encoding new sentences
    "tags": tags   # to map predictions back to labels
}, "../model/lstm_chatbot.pth")