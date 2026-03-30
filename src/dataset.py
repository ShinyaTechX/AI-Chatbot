# Dataset -> base class from PyTorch
# You inherit from it to create your own dataset

import torch
from torch.utils.data import Dataset

class ChatDataset(Dataset):   # custom dataset for chatbot
    def __init__(self, X, y):   # This runs when create the dataset. store data
        self.n_samples = len(X)   # Total number of samples. Example: 100 sentences -> n_samples = 100
        self.x_data = X   # x_data -> input data(bag of words)
        self.y_data = y   # y_data -> labels(tags as numbers)
    
    def __getitem__(self, index):   # How to get ONE sample from dataset. get one sample
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):    # Returs total dataset size. number of samples
        return self.n_samples