# Import list

import numpy as np
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelBinarizer
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import ReduceLROnPlateau
import nltk
from nltk.tokenize import word_tokenize

import gradio as gr

import functions as func
import constants as const
import controlVariables as convar
from NextWordPredictorModel import NextWordPredictor

text = func.read_text_list(const.PATHS)

text = text.lower()

no_cover_text = func.clean_cover_text_str(text, convar.clean_cover)
text_divided = func.divide_text_str(no_cover_text, convar.dividers)

vocabulary, word_to_idx = func.tokenize_text(func.clean_text(no_cover_text, const.CLEAN_CHARS))
total_words = len(vocabulary) + 1

input_sequences = func.n_gram_separation(text_divided, const.CLEAN_CHARS, word_to_idx)

# Get the max value to add padding to other entries
input_seq_pad = func.pad_sequences(input_sequences)

# Metrics printing
average = 0
for seq in input_sequences:
    average += len(seq)
max_sequence_len = max([len(seq) for seq in input_sequences])

X, y = func.split_xy(input_seq_pad, total_words, convar.gpu_running)

dataset = const.TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

if convar.gpu_running:
    model = NextWordPredictor(
        vocab_size = total_words, 
        embed_dim = convar.embed_dim, 
        hidden_dim = convar.hidden_dim, 
        output_dim = total_words, 
        lstm_layers=convar.lstm_layers, 
        dropout=convar.dropout,
        temperature=convar.temperature
        ).cuda()
else:
    model = NextWordPredictor(
        vocab_size = total_words, 
        embed_dim = convar.embed_dim, 
        hidden_dim = convar.hidden_dim, 
        output_dim = total_words, 
        lstm_layers=convar.lstm_layers, 
        dropout=convar.dropout,
        temperature=convar.temperature
        )

criterion = convar.criterion
optimizer = optim.Adam(model.parameters(), lr=convar.lr)

if convar.gpu_running:
  print(f"Is coda avialable: {torch.cuda.is_available()}")

better_model = torch.load('/teamspace/studios/this_studio/model_test.pt', map_location=torch.device('cpu'))

# Predict text
if convar.gpu_running:
    device = torch.device('cpu')
    better_model.to(device)  # Asegurarse de que el modelo esté en el dispositivo correcto

#seed_text = func.predict_model (better_model, convar.next_words, convar.seed_text, word_to_idx, max_sequence_len)

#print(seed_text)

# Define a wrapper function to interface with Gradio
def gradio_predict(seed_text, next_words):
    return func.predict_model(better_model, next_words, seed_text, word_to_idx, max_sequence_len)


# Create the Gradio interface
with gr.Blocks() as demo:
    seed_text = gr.Textbox(lines=2, placeholder="Enter seed text here...", label="Seed Text")
    next_words = gr.Number(default=5, label="Number of words to predict")
    output = gr.Textbox(label="Generated Text")
    
    submit_button = gr.Button("Generate")
    
    submit_button.click(fn=gradio_predict, inputs=[seed_text, next_words], outputs=output)

# Launch the interface
demo.launch()