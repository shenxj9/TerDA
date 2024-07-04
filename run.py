import argparse
import logging
from numpy import arange
from numpy import argmax
import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model.LoadData import loaddata, generated_matrix
from torch.optim import lr_scheduler
from model.Model import Model
from utils import process_data, TrainDataset, TestDataset,to_triplets,NegDataset
import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score,precision_recall_curve,auc,f1_score
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
def lr_epoch(epoch):
    if epoch < 10:
        lr = 1
    else:
        lr = 0.99 ** (epoch - 10)
    return lr

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
        self.logger = logging.getLogger(__name__)

    def Save_root (self):
        now_time = datetime.now().strftime('%y_%m_%d__%H_%M')
        root = self.prj_path /'output'
        if not root.exists():
            root.mkdir()
        save_root = root / f"{now_time}"
        self.models_path = save_root/'models'
        self.result_path = save_root /'results'
        if not save_root.exists():
            save_root.mkdir()
        if not self.models_path.exists():
            self.models_path.mkdir()
        if not self.result_path.exists():
            self.result_path.mkdir()

    def Get_triplets(self, train_pos_idx, test_pos_idx, train_neg_idx, test_neg_idx):
        train_data, test_data = self.df_data[train_pos_idx], self.df_data[test_pos_idx]
        self.train_neg, self.test_neg = self.df_negs[train_neg_idx], self.df_negs[test_neg_idx]
        self.train_data = np.asarray(to_triplets(train_data, self.entity_dict))
        self.test_data = np.asarray(to_triplets(test_data, self.entity_dict))
        self.labels = generated_matrix(self.p.num_drugs, self.p.num_diseases, self.train_neg, self.test_neg,
                                       self.train_data)
        self.triplets = process_data({'train': self.train_data, 'test': self.test_data, 'test_neg': self.test_neg})


    def Pos_Neg_dict(self):
        self.pos_dict = defaultdict(list)
        for i in range(self.p.num_drugs):
            index = [j for j in range(self.p.num_diseases) if self.labels[i][j] == 2]
            self.pos_dict[i].extend([len(index), index])

        self.neg_dict = defaultdict(list)
        for i in range(self.p.num_drugs):
            index = [j for j in range(self.p.num_diseases) if self.labels[i][j] == 1]
            self.neg_dict[i].extend([len(index), index])

    def restore_saved_model(self):
        if self.p.restore:
            self.logger.info('Successfully Loaded previous model')
            self.load_model(self.p.model_saved_path)
        else:
            self.saved_epoch = 0

    def Start(self):
        self.Save_root()
        epoch_list = []
        loss_epoch = []
        val_results_list = []
        nfold_list = []
        kf = KFold(n_splits=self.p.nfold, shuffle=True, random_state=200)
        nfold = 0
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(self.df_data),
                                                                                kf.split(self.df_negs)):
            print(nfold, 'nfold*************************')
            nfold = nfold + 1
            self.Get_triplets(train_pos_idx, test_pos_idx, train_neg_idx, test_neg_idx)
            self.Pos_Neg_dict()
            self.data_iter = self.get_data_iter()
            self.model= self.get_model()
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: lr_epoch(epoch))
            self.restore_saved_model()
            for epoch in range(self.saved_epoch,self.saved_epoch+self.p.max_epochs):
                print('Time:',datetime.now().strftime('%H:%M'),'epoch:',epoch)
                train_loss = self.train()
                nfold_list.append(nfold)
                epoch_list.append(epoch)
                loss_epoch.append(train_loss)
                self.scheduler.step()
                val_results = self.predict(epoch)
                self.logger.info(
                    f"[Epoch {epoch+1}]: lr:{self.optimizer.state_dict()['param_groups'][0]['lr']:.4},Training Loss: {train_loss:.4}, MR: {val_results['mr']:.4} MRR: {val_results['mrr']:.4},"
                    f"'roc_auc':{val_results['roc_auc']:.3},'ACC':{val_results['accuracy']:.3},'pr_auc':{val_results['pr_auc']:.3}")
                val_results['nfold'] = nfold
                val_results['epoch'] = epoch + 1
                val_results = pd.DataFrame(val_results, index=[0])
                val_results_list.append(val_results)

            df = {"nfold": nfold_list,"epoch": epoch_list,'loss': loss_epoch, }
            df = pd.core.frame.DataFrame(df)
            save_path = self.result_path/f'train_result_{nfold}_{epoch+1}.csv'
            df.to_csv(save_path, sep='\t',index=False)
            val_results_output = pd.concat(val_results_list)
            save_path = self.result_path/f'valid_result_{nfold}_{epoch+1}.csv'
            val_results_output.to_csv(save_path, sep='\t',index=False)

    def get_target_pred(self,subj, pred):
        train_pos = []
        train_neg = []
        for m, n in enumerate(subj.tolist()):
            train_pos.extend([[m, j] for j in self.pos_dict[n][1]])
            train_neg.extend([[m, j] for j in self.neg_dict[n][1]])
        index_i = torch.hstack((torch.tensor(train_pos)[:, 0], torch.tensor(train_neg)[:, 0]))
        index_j = torch.hstack((torch.tensor(train_pos)[:, 1], torch.tensor(train_neg)[:, 1]))
        target_pred = pred[index_i, index_j]
        labels = torch.hstack((torch.ones(len(train_pos)), torch.zeros(len(train_neg))))
        return target_pred, labels

    def train(self):
        self.model.train()
        losses = []
        train_iter = self.data_iter['train']
        for step, (triplets, labels) in enumerate(train_iter):
            if self.p.gpu >= 0:
                triplets= triplets.to("cuda")
            subj, rel = triplets[:, 0], triplets[:, 1]
            pred, _= self.model(subj, rel)  # [batch_size, num_ent] #torch.Size([256, 22272])
            target_pred, labels = self.get_target_pred(subj, pred)
            loss = self.model.calc_loss(target_pred, labels.to("cuda"))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        losses = [round(i, 5) for i in losses]
        loss = np.mean(losses)
        return loss

    def predict_pos(self,epoch):
        pos_preds = []
        rank_pred = []
        test_iter = self.data_iter['test']
        for step, (triplets, labels) in enumerate(test_iter):
            triplets, labels = triplets.to("cuda"), labels.to("cuda")
            subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2] - self.p.num_drugs
            _, pred = self.model(subj, rel)
            b_range = torch.arange(pred.shape[0], device="cuda")
            target_pred = pred[b_range, obj]
            pred = torch.where(
                labels.bool(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, obj] = target_pred
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]
            pos_preds.extend(target_pred.tolist())
            rank_pred.extend(ranks.tolist())
        return pos_preds, rank_pred

    def predict_neg(self):
        neg_preds = []
        test_iter = self.data_iter['test_neg']
        for step, (triplets, labels) in enumerate(test_iter):
            triplets, labels = triplets.to("cuda"), labels.to("cuda")
            subj, rel = triplets[:, 0], triplets[:, 1]
            _, pred = self.model(subj, rel)
            target_pred = pred[labels.bool()]
            neg_preds.extend(target_pred.tolist())
        return neg_preds

    def get_calssification_metrics(self, results, labels, pred):
        results['roc_auc'] = roc_auc_score(labels, pred)
        precision, recall, thresholds = precision_recall_curve(labels, pred, pos_label=1)
        results['pr_auc'] = auc(recall, precision)
        thresholds = arange(0.2, 0.8, 0.1)
        scores = [f1_score(labels, self.to_label(pred, t)) for t in thresholds]
        ix = argmax(scores)
        best_threshold = thresholds[ix]
        target_pred_labels = np.array([0 if i < best_threshold else 1 for i in pred])
        results['accuracy'] = accuracy_score(labels, target_pred_labels)
        results['precision'] = precision_score(labels, target_pred_labels)
        results['recall'] = recall_score(labels, target_pred_labels)
        return results

    def get_rank_metrics(self, results, rank_pred):
        ranks = torch.tensor([rank_pred])
        count = torch.numel(ranks)
        ranks = ranks.float()
        results['ranks'] = [ranks.cpu().tolist()]
        results['mr'] = round(torch.sum(ranks).item() / count, 5)
        results['mrr'] = round(torch.sum(1.0 / ranks).item() / count, 5)
        for k in [1, 3, 10]:
            results[f'hits@{k}'] = round(torch.numel(
                ranks[ranks <= k]) / count, 5)
        return results

    def get_metrics(self, results, labels, pred, rank_pred):
        results = self.get_calssification_metrics(results, labels, pred)
        results = self.get_rank_metrics(results, rank_pred)
        return results

    def predict(self,epoch):
        self.model.eval()
        with torch.no_grad():
            results = dict()
            pos_preds, rank_pred = self.predict_pos(epoch)
            neg_preds = self.predict_neg()
            all_pred = torch.tensor(pos_preds+neg_preds)
            labels = torch.hstack((torch.ones(len(pos_preds)), torch.zeros(len(neg_preds))))
            results = self.get_metrics(results, labels, all_pred, rank_pred)
        return results

    def to_label(self, x, t):
        return np.array([0 if i < t else 1 for i in x])

    def save_model(self, path,epoch):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
        :param path: path where the model is saved
        :return:
        """
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p),
            'saved_epoch': epoch + 1,
        }
        torch.save(state, path)

    def load_model(self, path,):
        """
        Function to load a saved model
        :param path: path where model is loaded
        :return:
        """
        state = torch.load(path)
        self.model.load_state_dict(state['model'],False)
        self.optimizer.load_state_dict(state['optimizer'])
        self.saved_epoch = state['saved_epoch']

    def get_data_iter(self):
        """
        get data loader for train, valid and Repositioning_all section
        :return: dict
        """

        def get_data_loader(dataset_class, split):
            return DataLoader(
                dataset_class(self.triplets[split], self.p.num_diseases,self.p),
                batch_size=self.p.batch_size,
                shuffle=True,
                num_workers=self.p.num_workers
            )

        return {
            'train': get_data_loader(TrainDataset, 'train'),
            'test': get_data_loader(TestDataset, 'test'),
            'test_neg': get_data_loader(NegDataset, 'test_neg'),
        }

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
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', dest='max_epochs',
                        type=int, default=300, help='Number of epochs')
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

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    runner = Runner(args)
    runner.Start()
