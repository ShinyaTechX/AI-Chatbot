# True LSTM Chatbot(Sequence Learning)

A deep learning chatbot built with PyTorch using a true LSTM sequence model(not bag-of-words).
This project demonstrates how to train a chatbot that understands word order and context using sequence learning.

## Features
- True LSTM-based sequence modeling
- Understands word order(context-aware)
- Custom tokenization + stemming pipeline
- Padding & batching for sequence input
- Intent classification with CrossEntropyLoss
- Easy deployment(CLI/Telegram bot)
- Clean and minimal architecture for learning

## Model Architecture
Input -> LSTM -> Fully Connected -> Softmax -> Intent
- Input size: 1(word index per timestep)
- Hidden size: 8
- Output size: Number of intents
- Loss: CrossEntropyLoss
- Optimizer: Adam

## Data Pipeline
- Tokenize sentence
- Apply stemming
- Convert words -> indices(word2idx)
- Pad sequences to fixed length
- Reshape -> (batch, seq_len, input_size)
- Feed into LSTM

## Run
python train_lstm.py
python chat_lstm.py