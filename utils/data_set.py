from torch.utils.data import Dataset
import numpy as np
import torch


class TrainDataset(Dataset):
    def __init__(self, triplets, num_diseases, params, ):
        super(TrainDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.num_diseases = num_diseases

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.num_diseases], dtype=np.float32)
        y[label-self.p.num_drugs] = 1
        return torch.tensor(y, dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, triplets, num_diseases,params):
        super(TestDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.num_diseases = num_diseases

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.num_diseases], dtype=np.float32)
        y[label-self.p.num_drugs] = 1
        return torch.tensor(y, dtype=torch.float32)


class NegDataset(Dataset):
    def __init__(self, triplets, num_diseases,params):
        super(NegDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.num_diseases = num_diseases

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        return triple, label

    def get_label(self, label):

        y = np.zeros([self.num_diseases], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)