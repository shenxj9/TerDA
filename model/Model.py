import torch
from torch import nn
from torch.nn import functional as F
from dgl.nn.pytorch import GraphConv
from dgllife.model.readout.mlp_readout import MLPNodeReadout
from model.atom_feature import get_graphs_features
import numpy as np

def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param

class BaseModel(nn.Module):
    def __init__(self, smi_dict, description_dict,params):
        super(BaseModel, self).__init__()
        self.p = params
        self.smi_dict = smi_dict
        self.description_dict = description_dict
        self.init_embed = get_param((self.p.num_drugs, self.p.init_dim))
        self.dieases_embed = get_param((self.p.num_diseases, self.p.init_dim))
        self.init_rel = get_param((1, self.p.init_dim))
        self.bias = nn.Parameter(torch.zeros(self.p.num_diseases))
        self.drug_gcn = torch.nn.ModuleList([GraphConv(in_feats=78, out_feats=self.p.init_dim), GraphConv(in_feats=self.p.init_dim, out_feats=self.p.init_dim)])
        self.drug_output_layer = MLPNodeReadout(self.p.init_dim, self.p.init_dim, self.p.init_dim,
                                                activation=nn.ReLU(), mode='max')
        self.drug_embedding = nn.Embedding(self.p.vocab_size, self.p.drug_embedding_dim, padding_idx=0)
        self.drug_cnn_layers = nn.ModuleList([nn.Conv1d(in_channels=self.p.drug_embedding_dim,
                                                        out_channels=self.p.init_dim, kernel_size=3, padding=1)])
        self.fc1 = nn.Linear(2*self.p.init_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.p.init_dim)
        self.relu = nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(self.p.init_dim)
        self.Tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.p.neg_ratio))
        # self.w = nn.Parameter(torch.ones(2))
        self.hyper_emb = get_param((1, self.p.init_dim))

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def compound_feature(self, sub):
        smi_batch = [self.smi_dict[key] for key in sub.tolist()]
        compound_graphs, compound_features = get_graphs_features(smi_batch)  # compound_features: torch.Size([6693, 78])
        compound_graphs = compound_graphs.to("cuda")
        compound_vector = compound_features.clone().detach().to("cuda")
        for l in self.drug_gcn:
            compound_vector = F.relu(l(compound_graphs, compound_vector))
        compound_vector = self.drug_output_layer(compound_graphs, compound_vector)
        compound_vector = self.bn(compound_vector)

        self.description_batch = torch.from_numpy(np.stack([self.description_dict[key][1] for key in sub.tolist()])).long().cuda()
        drugs_input = self.drug_embedding(self.description_batch)
        drugs_input = drugs_input.permute(0, 2, 1)
        for l in self.drug_cnn_layers:
            drugs_input = self.Tanh(self.bn(F.adaptive_max_pool1d((l(drugs_input)),output_size=1)))
        description_vector = drugs_input.view(drugs_input.size(0), -1)
        vector = torch.cat((compound_vector, description_vector), 1)
        vector = self.fc1(vector)
        vector = self.relu(vector)
        vector = self.dropout(vector)
        vector = self.fc2(vector)
        vector = self.relu(vector)
        vector = self.dropout(vector)
        vector = self.fc3(vector)
        return vector

    def entity_embed(self, compound_vector, sub, rel):
        r = self.init_rel
        rel_emb = torch.index_select(r, 0, rel)
        x = self.init_embed
        entity_emb = torch.index_select(x, 0, sub)
        sub_emb = entity_emb * 0.8 + compound_vector * 0.2
        return rel_emb, sub_emb

    def DistMult(self,sub_emb, rel_emb):
        obj_emb = sub_emb  * rel_emb
        pred = torch.mm(obj_emb, self.dieases_embed.transpose(1, 0))
        return pred


class Model(BaseModel):
    def __init__(self, smi_dict, description_dict,params):
        super(self.__class__, self).__init__(smi_dict,description_dict,params)
    def forward(self, sub , rel):
        compound_vector = self.compound_feature(sub)
        rel_emb, sub_emb = self.entity_embed(compound_vector, sub, rel)
        pred = self.DistMult(sub_emb, rel_emb)
        pred += self.bias.expand_as(pred)
        score = torch.sigmoid(pred)
        return pred, score
