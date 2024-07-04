import numpy as np
import os
import pandas as pd
import random
np.random.seed(123)
import torch
def loaddata(p):
    data = Dataset(p)
    data.load()
    return data

class Dataset(object):
    def __init__(self, p):
        self.dir = f'./dataset/'
        self.neg_ratio = p.neg_ratio
        self.num_drugs = p.num_drugs
        self.num_diseases = p.num_diseases
        self.mol_path =  p.mol_path
        self.smi_dict = {}
        self.num_diseases = p.num_diseases
        self.description_path = p.description_path
    def load(self):
        entity_path = os.path.join(self.dir, 'entityid.csv')
        data = os.path.join(self.dir, 'kg.csv')
        df = pd.read_csv(data,header = None,sep = '\t')
        self.data = df.values
        self.entity_dict = _read_dictionary(entity_path)
        print('len(self.entity_dict)',len(self.entity_dict))
        self.num_nodes = len(self.entity_dict)
        self.matrix = np.zeros((self.num_drugs, self.num_diseases), dtype=np.int32)

        for i in range(len(df)):
            a = self.entity_dict[df.loc[i][0]]
            b = self.entity_dict[df.loc[i][2]]
            self.matrix[a, b - self.num_drugs] = 1

        self.negs = self.generated_neg()
        if self.mol_path is not None:
            self.smi_dict= self.mol_feature(self.mol_path)
            self.smi_dict = sorted(self.smi_dict.items(), key=lambda x: x[0])


        if self.description_path is not None:
            self.description_dict= self.description_feature(self.description_path)
            self.description_dict = sorted(self.description_dict.items(), key=lambda x: x[0])


    def generated_neg(self):
        negs = dict()
        for i in range(self.num_drugs):
            neg_index = [j for j in range(self.num_diseases) if self.matrix[i][j] == 0]
            num = int(sum(self.matrix[i]).item()) * self.neg_ratio
            if len(neg_index) >= num :
                index = random.sample(neg_index, num)
            else:
                index = neg_index
            negs[i] = index

        negs1 = []
        for i in range(self.num_drugs):
            for j in negs[i]:
                negs1.append([i, j])
        return negs1

    def description_feature(self,mol_path):
        description_dict = dict()
        mol_path = os.path.join(self.dir, mol_path)
        df = pd.read_csv(mol_path, sep='\t', header = 0)
        # self.description_vector = df.values[:,2:].astype(int)
        for i in range(len(df)):
            id = self.entity_dict[df.iloc[i]['id']]
            description_dict[id] = np.array(df.iloc[i][2:]).astype(int)
        return description_dict


    def mol_feature(self,mol_path):
        mol_path = os.path.join(self.dir, mol_path)
        df = pd.read_csv(mol_path, sep='\t', header = None)
        # self.description_vector = df.values[:,2:].astype(int)
        for i in range(len(df)):
            id = self.entity_dict[df.iloc[i][0]]
            self.smi_dict[id] = df.loc[i][1]
        return self.smi_dict

def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d

def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line

def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l

def generated_matrix(num_drugs, num_diseases, train_neg, test_neg, train_pos):
    matrix = np.zeros((num_drugs, num_diseases), dtype=np.int32)
    for i in range(len(train_neg)):
        matrix[train_neg[i][0], train_neg[i][1]] = 1
    for i in range(len(test_neg)):
        matrix[test_neg[i][0], test_neg[i][1]] = -1
    for i in range(len(train_pos)):
        matrix[train_pos[i][0], train_pos[i][2]-num_drugs] = 2
    return matrix

