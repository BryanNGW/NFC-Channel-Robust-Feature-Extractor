import mat73
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, Sampler
import glob 
import numpy as np
import os
import re
#from PIL import Image
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#from torchvision.transforms import transforms
from torch.utils.data import Dataset
#from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
from collections import defaultdict
from scipy.io import loadmat

def sort_key(file_name):     
        number = re.search(r'\d+', file_name)
        if number:
            return int(number.group())


class CISDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.samples = []

        # Traverse the directory structure to collect file paths, labels, and distance folder names
        for first_layer_folder in sorted(os.listdir(data_root), key=sort_key):
            first_layer_path = os.path.join(data_root, first_layer_folder)
            if os.path.isdir(first_layer_path):
                for card_idx, card_folder in enumerate(sorted(os.listdir(first_layer_path), key=sort_key)):
                    card_path = os.path.join(first_layer_path, card_folder)
                    if os.path.isdir(card_path):
                        for file in sorted(os.listdir(card_path), key=sort_key):
                            if file.endswith('.mat'):
                                file_path = os.path.join(card_path, file)
                                # Append a tuple with three elements: file path, card label, and distance folder
                                self.samples.append((file_path, card_idx, first_layer_folder))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Handle cases with 2 or 3 elements in the tuple
        sample = self.samples[idx]
        if len(sample) == 3:
            CIS_path, label, distance_folder = sample
        elif len(sample) == 2:
            CIS_path, label = sample
            distance_folder = None  # Assign a default value if the distance folder is missing
        else:
            raise ValueError(f"Unexpected sample format: {sample}")

        # Load data from the .mat file
        try:
            # Load using mat73 for MATLAB 7.3 files
            CIS = mat73.loadmat(CIS_path, use_attrdict=True)['sample_data']
        except (TypeError, KeyError):
            # Fallback to scipy.io.loadmat for older MATLAB files
            mat_data = loadmat(CIS_path)
            if 'sample_data' in mat_data:
                CIS = mat_data['sample_data']
            else:
                raise KeyError(f"Key 'sample_data' not found in {CIS_path}. Available keys: {list(mat_data.keys())}")

        # Convert to Torch tensor and add a new dimension
        CIS = torch.tensor(CIS).unsqueeze(0)

        return CIS, label, distance_folder    

class RxAgnosticCISDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.samples = []

        for distance_folder in os.listdir(self.data_root):
            distance_path = os.path.join(self.data_root, distance_folder)
            if not os.path.isdir(distance_path):
                continue

            # Traverse the card subfolders, which are the class labels
            for card_folder in os.listdir(distance_path):
                card_path = os.path.join(distance_path, card_folder)
                for file in os.listdir(card_path):
                    if file.endswith('.mat'):
                        file_path = os.path.join(card_path, file)
                        self.samples.append((file_path, card_folder, distance_folder))  # Three elements in the tuple

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Unpack file path and metadata
        CIS_path, label, distance_folder = self.samples[idx]
        try:
            mat_data = mat73.loadmat(CIS_path, use_attrdict=True)
        except Exception:
            mat_data = loadmat(CIS_path)
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if len(keys) != 1:
            raise ValueError(f"Expected 1 data variable in {CIS_path}, found {len(keys)}: {keys}")
        # Extract matrix (either 'S_out' or 'Q_out')
        CIS = mat_data[keys[0]]
        CIS = torch.tensor(CIS, dtype=torch.float32).unsqueeze(0)
        return CIS, label, distance_folder