# Full flow
# Input(batch, seq_len, input_size) -> LSTM(process sequence step by step) -> Output(batch, seq_len, hidden_size) -> Take last time step -> (batch, hidden_size) -> Linear layer -> (batch, output_size)

# Intuition
# Think of LSTM like reading a sentence: "I love machine learning":
# Reads word by word
# Remembers important info
# final state = understanding of whole sentence
# Then fc layer -> decides class(intent)
# Why take last output? Because it represents: "everything the model learned from the whole sequence"

# Summary
# LSTM reads sequence step-by-step
# Keeps memory(h, c)
# Outputs hidden states for each time step
# You take the last one
# Pass it to Linear layer -> prediction

# Big idea of LSTM
# LSTM = Long Short-Term Memory
# It solves this problem:
# Normal neural nets -> forget past. RNN -> struggle with long memory. LSTM -> remembers important things for long time

# What happens at ONE time step
# At each time step(word), LSTM decides:
# 1. What to forget
# 2. What to add
# 3. What to output

# torch -> core PyTorch library
# nn -> neural network module(layers like LSTM, Linear, etc.)

import torch
import torch.nn as nn

class LSTMModel(nn.Module):   # creating a custom neural network model. Inherits from nn.Module. This is required for all PyTorch models
    def __init__(self, input_size, hidden_size, output_size): # build the model
        super(LSTMModel, self).__init__()   # Initializes the parent class. Required so PyTorch can track layers & parameters
        self.hidden_size = hidden_size   # Save hidden size. Needed later to create hidden states
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)   # LSTM layer. input_size -> features per time step, hidden_size -> size of LSTM memory, batch_first = True -> input shape: (batch, seq_len, input_size)
        self.fc = nn.Linear(hidden_size, output_size)   # Fully connected layer. Converts LSTM output -> final prediction. Input: hidden_size, Output: output_size(number of classes)
    
    def forward(self, x):   # how data flows: This defines how input -> output
        # x shape: (batch, seq_len, input_size). Example: (8, 10, 20): 8 samples, 10 time steps, 20 features
        h0 = torch.zeros(1, x.size(0), self.hidden_size)   # Initialize hidden state(h0). 1 -> number of LSTM layers, x.size(0) -> batch size, hidden_size -> memory size. This is the initial hidden memory
        c0 = torch.zeros(1, x.size(0), self.hidden_size)   # Initialize cell state(c0). LSTM has TWO states: h: Short-term memory, c: long-term memory
        
        out, _ = self.lstm(x, (h0, c0))   # Pass through LSTM. LSTM processes sequence step by step. Updates memory(h, c) at each time step. Output: out.shape = (batch, seq_len, hidden_size). Example: (8, 10, 64). For each time step, we get a hidden output

        # Take last output
        out = out[:, -1, :]   # Take last time step. : -> all batches, -1 -> last time step, : -> all features. Result shape: (batch, hidden_size). We only use the final memory of the sequence
        out = self.fc(out)   # Fully connected layer. Converts hidden state -> prediction. Shape becomes: (batch, output_size)

        return out   # Return output. Final result = raw scores(logits)