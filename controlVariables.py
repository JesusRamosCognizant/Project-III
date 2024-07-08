""" File containing all the control variables """

import torch
import torch.nn as nn
import constants as const

# Text path
const.PATH_U = ""

# Determine device on which to run, True -> gpu; False -> cpu
gpu_running = False

# Variables for division of text
dividers = const.DIVIDERS_ORIGINAL
clean_cover = False

# Model parameters
embed_dim = 200
hidden_dim = 150
lstm_layers=1
dropout=0.2
temperature=1
# Model parameters
criterion = nn.CrossEntropyLoss()


# Training variables 
epochs = 500
patience = 20  # Number of epochs to wait for improvement
lr = 0.001

save_model = True  #If set to True, saves a torch_model.pt 

# Predict values
seed_text = "I am"
next_words = 10
