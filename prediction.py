import argparse
import logging
import numpy as np
import random
from pathlib import Path
import torch
from model.LoadData import loaddata
from model.Model import Model
import os
import pandas as pd

class Runner(object):
    def __init__(self, params):
        self.p = params
        self.prj_path = Path(__file__).parent.resolve()
        self.data = loaddata(self.p)
        self.num_ent,self.entity_dict = self.data.num_nodes,self.data.entity_dict
        self.df_negs= np.array(self.data.negs)
        self.df_data = self.data.data
        self.smi_dict = self.data.smi_dict
        self.description_dict = self.data.description_dict
        self.reversed_entity_dict = {v: k for k, v in self.entity_dict.items()}

        all_data = os.path.join(f'./dataset/assay_compound_info.csv')
        all_df = pd.read_csv(all_data, header=None, sep='\t')
        self.compounds = all_df[0].tolist()
        self.sub = torch.tensor([self.entity_dict[i] for i in self.compounds])



        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)


    def predict(self):
        self.model = self.get_model()
        self.model_saved_path = [
            r'saved_models/time_06_29__16_32.pt',
            r'saved_models/time_06_29__01_34.pt',
            r'saved_models/time_06_29__09_00.pt',
            r'saved_models/time_06_29__12_46.pt',
            r'saved_models/time_06_11__15_50.pt',
            r'saved_models/time_06_11__16_09.pt']

        preds_list = []
        for path in self.model_saved_path:
            self.load_model(path)
            self.model.eval()
            with torch.no_grad():
                subj = self.sub.to("cuda")
                rel = torch.tensor([0]).to("cuda")
                _, preds = self.model(subj, rel)
                preds_list.append(preds)

        preds = torch.stack(preds_list, dim=0).mean(dim=0)
        disease_ids = list(self.entity_dict.keys())[self.p.num_drugs:]
        preds_df = pd.DataFrame(preds.tolist())
        preds_df.columns = disease_ids
        preds_df.index = self.compounds

        ranks = 1 + torch.argsort(torch.argsort(preds, dim=1, descending=True), dim=1, descending=False)
        ranks_df = pd.DataFrame(ranks.tolist())
        ranks_df.columns = disease_ids
        ranks_df.index = self.compounds
        return preds_df[self.p.quary_disease], ranks_df[self.p.quary_disease]


    def load_model(self, path,):
        state = torch.load(path)
        self.model.load_state_dict(state['model'],False)

    def get_model(self):
        model = Model(smi_dict = self.smi_dict, description_dict = self.description_dict, params=self.p)
        if self.p.gpu >= 0:
            model.to("cuda")
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mol_path', dest='mol_path',default='drug_dict.csv')
    parser.add_argument('--description_path', dest='description_path', default='description.csv')
    parser.add_argument('--num_drugs', dest='num_drugs', default=6334)
    parser.add_argument('--num_diseases', dest='num_diseases', default=747)
    parser.add_argument('--vocab_size', dest='vocab_size', default=41)
    parser.add_argument('--nfold', dest='nfold', default=5,type=int,
                        help='Dataset to use,')
    parser.add_argument('--neg_ratio', dest='neg_ratio', default=1,type=int,
                        help='Ratio of positive and negative interactions')
    parser.add_argument('--batch', dest='batch_size',
                        default=512, type=int, help='Batch size')
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--epoch', dest='max_epochs',
                        type=int, default=500, help='Number of epochs')
    parser.add_argument('--l2', type=float, default=0.0,
                        help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Starting Learning Rate')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=12345,
                        type=int, help='Seed for randomization')
    parser.add_argument('--restore', dest='restore', action='store_true',
                        help='Restore from the previously saved model')
    parser.add_argument('--model_saved_path', dest='model_saved_path',
                        help='Path of the previously saved model')
    parser.add_argument('--bias', dest='bias', action='store_true',default=True,
                        help='Whether to use bias in the model')
    parser.add_argument('--init_dim', dest='init_dim', default=256, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('--drug_embedding_dim', dest='drug_embedding_dim', type=int, default=64)
    parser.add_argument('--data', dest='dataset', default='20uM_4')
    parser.add_argument('--predicted_path', dest='predicted_path', default=None)
    parser.add_argument('--quary_disease', dest='quary_disease', default=['NCIt; C3224; Melanoma'])


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    runner = Runner(args)
    df_pred, df_rank = runner.predict()
    df_rank.to_csv(r'pred_result.csv', sep='\t')



