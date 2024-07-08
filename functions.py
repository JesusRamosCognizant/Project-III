''' File containing all the functions '''

import re
from nltk.tokenize import word_tokenize

import numpy as np
import torch
import torch.nn as nn


''' 
Function to read text from file in path: paths
Parameters:
    paths - List with all posible paths 
'''
def read_text_list (paths: list):
    text = ""
    # Try all the different paths
    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                text = file.read()
                print(path)
        except:
            continue
        else:
            break
    return text if text != "" else -1

''' 
Function to read text from file in path: path
Parameters:
    paths - Str with the path
'''
def read_text_str (path: str):
    return read_text_list ([path])


''' 
Function to divide to keep or delete the front page
Parameters:
    text - Str with the text to divide
    clean_cover - Control variable 
'''
def clean_cover_text_str (text: str, clean_cover: bool):
    if clean_cover:
        # Delete cover of book and extra information
        text = text[980:-550]

    return text


''' 
Function to divide the text into a list using the given dividers
Parameters:
    text - Str with the text to divide
    dividers - List of charaters the behave as a divider
'''
def divide_text_str (text: str, dividers: list):
    # Split following the dividers given
    text = re.split(dividers, text)

    return text


''' 
Function to delete the characters given in list: clean_chars from the text: text
Parameters:
    text - Str with the text to clean
    clean_chars - Str of charaters delete from text
'''
def clean_text (text: str, clean_chars: str):
    
    new_text = text.lower()
    for char in clean_chars:
        new_text = new_text.replace(char,"")

    new_text = new_text.replace("--"," ")
    
    return new_text
    

''' 
Function to delete the characters given in list: clean_chars from the text: text
Parameters:
    text - Str with the text to clean
    clean_chars - Str of charaters delete from text
'''
def tokenize_text (text: str):

    tokens = word_tokenize(text)
    vocabulary = set(tokens)

    word_to_idx = {word:idx for idx, word in enumerate(vocabulary)}

    return vocabulary, word_to_idx


''' 
Function to separate the text in different sequences of length n
Parameters:
    text_divided - List of sequences to tokenize and create n-grams
    clean_chars - Str of charaters delete from text
    word_to_idx - Dict with the tokens 
'''
def n_gram_separation (text_divided: list, clean_chars: str, word_to_idx: dict):
    input_sequences = []
    for line in text_divided:
        clean_line = clean_text(line, clean_chars)
        line_list = clean_line.split(' ')

        # Tokenize each sentence
        token_list = []
        for char in line_list:
            if char in word_to_idx.keys():
                token_list.append(word_to_idx[char])

        # Divide the different sentences in n-grams
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    return input_sequences


''' 
Function to separate add padding to the n-grams 
Parameters:
    input_sequences - List of n-grams to add padding
'''
def pad_sequences (input_sequences: list):

    max_sequence_len = max([len(seq) for seq in input_sequences])
    input_seq_pad = np.array([np.pad(seq, (max_sequence_len - len(seq), 0), mode='constant') for seq in input_sequences])

    return input_seq_pad


''' 
Function to separate the set into x and y and apply one hot encoding 
Parameters:
    input_seq_pad - List of data  
    total_words - Total number of different words 
'''
def split_xy (input_seq_pad: list, total_words: int, gpu_running: bool):
    X = input_seq_pad[:, :-1]
    y = input_seq_pad[:, -1]

    # Convert output to one-hot encoded vectors
    if gpu_running:
        y_tensor = torch.tensor(y, dtype=torch.int64)
        y = F.one_hot(y_tensor, num_classes=total_words)
    else:
        y = np.array(torch.nn.functional.one_hot(torch.tensor(y), num_classes=total_words))

    return X, y


''' 
Function to train model 
Parameters:
    model - model to train
    dataloader - data to train from
    criterion - criterion for training
    optimizer - optimizer of training
    epochs - Int with number of train cycles
    patience - Int with number of cycles till stop if no improvement
'''
def training_model (model, dataloader, criterion, optimizer, epochs: int, patience: int, gpu_running: bool):
    current_patience = patience
    best_loss = float('inf')  # Initialize best loss to a very high value
    better_model = model
    for epoch in range(epochs):
        # Training loop
        for i, (inputs, labels) in enumerate(dataloader):
            if gpu_running:
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()  # Mover datos a la GPU
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

    return better_model


''' 
Function to predict values with a model 
Parameters:
    better_model - model to be used to predict
    next_words - Int number of words to predict
    seed_text - Str to with word to predict
    word_to_idx - Dict with tokenicers
    max_sequence_len - Int with max length for padding
'''
def predict_model (better_model, next_words: int, seed_text: str, word_to_idx: dict, max_sequence_len: int):

    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    better_model.eval()  # Set the model to evaluation
    for _ in range(int(next_words)):
        tokens = word_tokenize(seed_text)
        token_list = [word_to_idx[word] for word in tokens if word in word_to_idx]
        token_list = np.pad(token_list, (max_sequence_len - len(token_list), 0), mode='constant')
        token_list = torch.tensor(token_list[-max_sequence_len:], dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            predicted = better_model(token_list).argmax(dim=1).item()

        output_word = idx_to_word[predicted]
        seed_text += " " + output_word

    return seed_text


