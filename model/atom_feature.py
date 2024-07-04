import numpy as np
import torch
from rdkit import Chem
import networkx as nx
from dgl import DGLGraph
import dgl
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smiles2graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None ,None ,None
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature/sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

def get_graphs_features(compound_smiles):
    h=list()
    graphs=list()
    for i in range(len(compound_smiles)):
        c_size, features, edge_index=smiles2graph(compound_smiles[i][1])
        g=DGLGraph()
        if c_size == None:
            print('compound_smiles:',compound_smiles[i][1])
        g.add_nodes(c_size)
        if edge_index:
            edge_index=np.array(edge_index)
            g.add_edges(edge_index[:,0],edge_index[:,1])

        for f in features:
            h.append(f)
        g.ndata['x']=torch.from_numpy(np.array(features))
        g=dgl.add_self_loop(g)
        graphs.append(g)
    g=dgl.batch(graphs)
    return g,torch.from_numpy(np.array(h,dtype=np.float32))

