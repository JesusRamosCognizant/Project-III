import numpy as np
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelBinarizer
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import ReduceLROnPlateau
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Different Editors paths
PATH_A = "data/sherlock-holm.es_stories_plain-text_advs.txt"
PATH_E = ""
PATH_G = "0. Projects/3/Project-III/data/sherlock-holm.es_stories_plain-text_advs.txt"
PATH_J = ""
PATH_M = "Project-III/data/sherlock-holm.es_stories_plain-text_advs.txt"

PATHS = [PATH_A, PATH_E, PATH_G, PATH_J, PATH_M]

text = ""

for path in PATHS:
    try:
        # Read the text file
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
            print(path)
    except:
        continue
    else:
        break

# Divide with regular expressions
DIVIDERS_ORIGINAL = "\n"
DIVIDERS_ALL = "[,.!?:;\"]|\n\n|--| and | that | which "
DIVIDERS_MIN = "[.!]|\n\n"
DIVIDERS_BAL = "[,.!?]|\n\n|--"
CLEAR_COVER = False

text_try = text.lower()

if CLEAR_COVER:
    # Delete cover of book and extra information
    text_try = text[980:-550]

# Split following the dividers given
text_try = re.split(DIVIDERS_ORIGINAL, text_try)

# Delete all the new line comments
text_try = [el.replace('\n', '') for el in text_try]

# Create Tokenizer object in python
CLEAR_COVER
tokens = word_tokenize(text)
vocabulary = set(tokens)
total_words = len(vocabulary) + 1

word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

print(f"total_words: {total_words}")

# Create input-output sequences
input_sequences = []
for line in text_try:
    line_list = line.rstrip(",.;:").split(' ')

    # Tokenize each sentence
    token_list = []
    for char in line_list:
        if char in word_to_idx.keys():
            token_list.append(word_to_idx[char])

    # Divide the different sentences in n-grams
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

average = 0
for seq in input_sequences:
    average += len(seq)

max_sequence_len, value = max([(len(seq), seq) for seq in input_sequences])
input_seq_pad = np.array([np.pad(seq, (max_sequence_len - len(seq), 0), mode='constant') for seq in input_sequences])

# Split the sequences into input (X) and output (y)
X = input_seq_pad[:, :-1]
y = input_seq_pad[:, -1]

# Convert output to one-hot encoded vectors
y = np.array(torch.nn.functional.one_hot(torch.tensor(y), num_classes=total_words))

# Create a custom Dataset class
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the model
class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout=0.2, temperature=1, activation='relu'):
        super(NextWordPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_embed = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim * (max_sequence_len - 1), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.T = temperature

        # Choose the activation function
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'leaky_relu':
            self.activation = lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        elif activation == 'elu':
            self.activation = torch.nn.functional.elu
        else:
            self.activation = torch.relu
#cambio
    def forward(self, sequences):
        embedded = self.embedding(sequences)
        embedded = self.dropout_embed(embedded)
        embedded = embedded.view(embedded.size(0), -1)
        hidden = self.activation(self.fc1(embedded))
        hidden = self.activation(self.fc2(hidden))
        logits = self.fc3(hidden)
        logits = self.softmax(logits / self.T)
        return logits

# You can change the activation function here
model = NextWordPredictor(vocab_size=total_words, embed_dim=100, hidden_dim=150, output_dim=total_words, temperature=3, activation='tanh')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

SAVE_MODEL = False  # If set to True, saves a torch_model.pt 
SAVED_MODEL_NAME = 'ME_CAGO_EN_VUESTRA_VIDA_GUARDAR_LOS_MODELOS_POR_FAVOR.pt'

epochs = 1
patience = 5  # Number of epochs to wait for improvement
current_patience = patience
best_loss = float('inf')  # Initialize best loss to a very high value
better_model = model
for epoch in range(epochs):
    # Training loop
    for i, (inputs, labels) in enumerate(dataloader):
        outputs = model(inputs)
        loss = criterion(outputs, labels.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Early stopping
    if loss.item() < best_loss:  # Compare current training loss with best loss
        best_loss = loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()} (Improved)')
        better_model = model
        current_patience = patience  # Restart patience
    else:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        current_patience -= 1  # Decrement patience counter on no improvement

    # Stop training if patience is 0
    if current_patience == 0:
        print('Early stopping triggered!')
        break

# Saving model to .pt
if SAVE_MODEL == True:
    torch.save(better_model, SAVED_MODEL_NAME)

# Initial text to predict
seed_text = "I am"
next_words = 10

# Index to word
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Generate the n next words
better_model.eval()  # Set the model to evaluation
for _ in range(next_words):
    tokens = word_tokenize(seed_text)
    token_list = [word_to_idx[word] for word in tokens if word in word_to_idx]
    token_list = np.pad(token_list, (max_sequence_len - len(token_list), 0), mode='constant')
    token_list = torch.tensor(token_list[-max_sequence_len:], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        predicted = better_model(token_list).argmax(dim=1).item()

    output_word = idx_to_word[predicted]
    seed_text += " " + output_word

print(seed_text)
