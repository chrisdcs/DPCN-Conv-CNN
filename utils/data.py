# customize pytorch data loader
import torch
from torch.utils.data import Dataset

class ZCA_Loader(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]