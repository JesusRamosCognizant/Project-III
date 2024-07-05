""" File containing the  model to run """

import torch
import torch.nn as nn


# Define the model
class NextWordPredictor(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, lstm_layers=1, dropout=0.2, temperature=1):
    super(NextWordPredictor, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.dropout_embed = nn.Dropout(dropout)  
    self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=lstm_layers, dropout=dropout, bidirectional=True, batch_first=True) 
    self.fc = nn.Linear(hidden_dim * 2, output_dim) 
    self.softmax = nn.Softmax(dim=1)
    self.T = temperature

  def forward(self, sequences):
    embedded = self.embedding(sequences)
    embedded = self.dropout_embed(embedded)
    lstm_out, _ = self.lstm(embedded)
    last_hidden = lstm_out[:, -1, :] 
    logits = self.fc(last_hidden)
    logits = self.softmax(logits/self.T) 
    return logits
