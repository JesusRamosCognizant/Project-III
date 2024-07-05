""" File containing all the constants """

from torch.utils.data import Dataset
import torch


# Different Editors paths
PATH_A = "data\sherlock-holm.es_stories_plain-text_advs.txt"
PATH_E = ""
PATH_G = "0. Projects/3/Project-III/data/sherlock-holm.es_stories_plain-text_advs.txt"
PATH_J = ""
PATH_M = "Project-III/data/sherlock-holm.es_stories_plain-text_advs.txt"
PATH_U = ""
MODEL_NAME = "torch_model.pt"

PATHS = [PATH_A, PATH_E, PATH_G, PATH_J, PATH_M, PATH_U]


# Different dividers for the text 
DIVIDERS_ORIGINAL = "\n"
DIVIDERS_ALL = "[,.!?:;\"]|\n\n|--| and | that | which "
DIVIDERS_MIN = "[.!]|\n\n"
DIVIDERS_BAL = "[,.!?]|\n\n|--"


# Chars to be deleted from the book
CLEAN_CHARS = ",.:;!?\n()[]'\""



# Create a custom Dataset class
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float)
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Text colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'