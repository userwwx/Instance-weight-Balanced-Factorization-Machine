from torch.utils.data import Dataset, DataLoader
import numpy as np


class CriteoDataset(Dataset):
    def __init__(self, xi, xv, y):
        self.xi = np.array(xi, dtype=np.long)
        self.xv = np.array(xv, dtype=np.float)
        self.y = np.array(y, dtype=np.float)

    def __getitem__(self, index):
        batch_xi = self.xi[index]
        batch_xv = self.xv[index]
        batch_y = self.y[index]
        return batch_xi, batch_xv, batch_y

    def __len__(self):
        return self.xi.shape[0]


class FrappeDataSet(Dataset):
    def __init__(self, data):
        self.xi = np.array(data['X'], dtype=np.long)
        self.label = np.array(data['Y'], dtype=np.float)

    def __getitem__(self, index):
        batch_xi = self.xi[index]
        batch_label = self.label[index]
        return batch_xi, batch_label

    def __len__(self):
        return self.xi.shape[0]

class AvazuDataSet(Dataset):
    def __init__(self, data):
        self.xi = np.array(data['index'], dtype=np.long)
        self.label = np.array(data['label'], dtype=np.float)
    
    def __getitem__(self, index):
        batch_xi = self.xi[index]
        batch_label = self.label[index]
        return batch_xi, batch_label
    
    def __len__(self):
        return self.xi.shape[0]