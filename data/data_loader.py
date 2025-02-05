# data/data_loader.py

from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles = dataframe['Drug'].tolist()
        self.labels = dataframe['Y'].values if 'Y' in dataframe.columns else None

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        label = self.labels[idx] if self.labels is not None else None
        return s, label