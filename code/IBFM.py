import sys
sys.path.append('/home/wwx/wwx/IBFM')
import argparse
from util.dataset import FrappeDataSet
from torch.utils.data import DataLoader
from util.LoadData import LoadData
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from time import time
from layers import MLPLayers, AttLayer

parser = argparse.ArgumentParser(description='Hyperparameter tuning')
parser.add_argument('-path', default='../data/', type=str, help='Input data path')
parser.add_argument('-dataset', default='frappe', type=str, help='Choose a dataset.')
parser.add_argument('-pretrain', default=1, type=int, help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize')
parser.add_argument('-epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('-metric', default='RMSE', type=str, help='MAE or RMSE')
parser.add_argument('-batch_size', default=2048, type=int, help='batich size')
parser.add_argument('-embedding_size', default=256, type=int, help='embedding size')
parser.add_argument('-field_size', default=10, type=int, help='the feature field of dataset')
parser.add_argument('-dnnlayers', default=[64, 64], type=list, help='the network structure of the instance-weight layer')
parser.add_argument('-l2', default=0.1, type=float, help='Regularizer for the instance-weight part')
parser.add_argument('-dnn_dropout', default=0.1, type=float, help='dropout rate for the instance-weight layer')
parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
parser.add_argument('-verbose', default=1, type=int, help='Whether to show the performance of each epoch (0 or 1)')
parser.add_argument('-attention_size', default=16, type=int, help='the attention size of the attention netwrok')
parser.add_argument('-la', default=32, type=float, help='regularizer for the attention network')
parser.add_argument('-att_dropout', default=0.3, type=float, help='dropout rate for the interaction attention layer')
parser.add_argument('-save', default=1, type=int, help='Whether to save the model state dict (1 or 0)')
pars = parser.parse_args()

if not torch.cuda.is_available():
    pars.use_cuda = 0

class IBFM(nn.Module):
    def __init__(self, features_M, field_size, embedding_size, epochs, batch_size,
                 metric_type, learning_rate, l2, att_dropout, dnn_dropout, dnnlayers, attention_size, la,
                pretrain, save_path, verbose, use_cuda, random_seed=2020):
        super(IBFM, self).__init__()
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self.att_dropout = att_dropout
        self.dnn_dropout = dnn_dropout
        self.dnnlayers = [self.field_size * self.embedding_size] + dnnlayers
        self.metric_type = metric_type
        self.verbose = verbose
        self.random_seed = random_seed
        self.features_M = features_M
        self.attention_size = attention_size
        self.la = la
        self.pretrain = pretrain
        self.save_path = save_path
        self.use_cuda = use_cuda
        self.num_pair = self.field_size * (self.field_size - 1) // 2

        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)

        self.bias = nn.Parameter(torch.tensor([.0]))
        self.fm_1st_embedding = nn.Embedding(self.features_M, 1)
        nn.init.xavier_normal_(self.fm_1st_embedding.weight)
        self.fm_2nd_embedding = nn.Embedding(self.features_M, self.embedding_size)
        nn.init.xavier_normal_(self.fm_2nd_embedding.weight)
        self.attlayer = AttLayer(self.embedding_size, self.attention_size)
        self.p = nn.Parameter(torch.randn(self.embedding_size))
        self.prediction = nn.Parameter(torch.randn(self.dnnlayers[-1], self.field_size))
        self.dropout_layer = nn.Dropout(self.att_dropout)
        self.dnn = MLPLayers(self.dnnlayers, bn=True, activation='relu', dropout=self.dnn_dropout, init_mothod='norm')

        # ---------------- load pretrain data ----------------------------------------
        if self.pretrain:
            b = np.load('../pretrain/bias.npy', 'r+')
            fm_1st = np.load('../pretrain/fm_1st.npy', 'r+')
            fm_2nd = np.load('../pretrain/fm_2nd.npy', 'r+')
            self.bias.data.copy_(torch.from_numpy(b))
            self.fm_1st_embedding.weight.data.copy_(torch.from_numpy(fm_1st))
            self.fm_2nd_embedding.weight.data.copy_(torch.from_numpy(fm_2nd))
        # ----------------------------------------------------------------------------


    def forward(self, xi):
        fm_1st_embedding = self.fm_1st_embedding(xi)
        fm_2nd_embedding = self.fm_2nd_embedding(xi)

        # --------------- instance weighting -----------------
        att = self.instance_weighting(fm_2nd_embedding)

        # --------------- weight balancing -------------------
        fm_2nd_embedding = torch.add(fm_2nd_embedding, torch.mul(fm_2nd_embedding, att))
        fm_1st_embedding = torch.add(fm_1st_embedding, torch.mul(fm_1st_embedding, att))
        
        # ---------------- prediction ----------------
        total_sum = torch.einsum('bnd->b', fm_1st_embedding) + self.interaction_attention(fm_2nd_embedding) + self.bias
        return total_sum

    def instance_weighting(self, fm_2nd_emb):
        att = self.dnn(fm_2nd_emb.reshape(-1, self.field_size * self.embedding_size))
        # ---------- sigmoid -----------------------------------
        att = torch.sigmoid(torch.matmul(att, self.prediction))
        # ------------------------------------------------------
        return att.unsqueeze(2)
    
        
    def interaction_attention(self, infeature):
        p, q = self.build_cross(infeature)
        pair_wise_inter = torch.mul(p, q)

        att_signal = self.attlayer(pair_wise_inter)
        att_signal = att_signal.unsqueeze(dim=2)

        att_inter = torch.mul(att_signal, pair_wise_inter)
        att_pooling = torch.sum(att_inter, dim=1)
        att_pooling = self.dropout_layer(att_pooling)


        att_pooling = torch.mul(att_pooling, self.p)
        att_pooling = torch.sum(att_pooling, dim=1)

        return att_pooling

    def build_cross(self, feat_emb):
        row = []
        col = []
        for i in range(self.field_size):
            for j in range(i + 1, self.field_size):
                row.append(i)
                col.append(j)
        p = feat_emb[:, row]
        q = feat_emb[:, col]
        return p, q

    def fit(self, train_data, valid_data, test_data):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate, weight_decay=0)
        criterion = nn.MSELoss(reduction='sum')
        if self.verbose:
            t2 = time()
            init_train = self.evaluate(train_data)
            init_valid = self.evaluate(valid_data)
            print("Init \t train=%.4f, validation=%.4f [%.1f s]" % (init_train, init_valid, time() - t2))

        for epoch in range(self.epochs):
            model = self.train()
            t1 = time()
            for (xi, y) in train_data:
                xi = xi.long()
                if self.use_cuda:
                    xi, y = xi.cuda(), y.cuda()
                optimizer.zero_grad()
                y_pred = model(xi)
                loss = criterion(y, y_pred)
                for name, param in model.named_parameters():
                    if 'weight' in name and 'dnn' in name and len(param.size()) == 2:
                        loss += model.l2 * torch.norm(param.data, 2)
                loss += (model.la * torch.norm(model.attlayer.w.weight, 2))
                loss += model.l2 * torch.norm(model.prediction.data, 2)
                loss.backward()
                optimizer.step()
            t2 = time()
            train_result = self.evaluate(train_data)
            valid_result = self.evaluate(valid_data)
            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\t train=%.4f, validation=%.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, train_result, valid_result, time() - t2))
            test_result = self.evaluate(test_data)
            self.test_rmse.append(test_result)
            print("Epoch %d [%.1f s]\t test=%.4f [%.1f s]"
                  % (epoch + 1, t2 - t1, test_result, time() - t2))

        if self.save_path != '':
            torch.save(model.state_dict(), save_path)

    def eva_termination(self, valid):
        if len(valid) > 5:
            if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                return True
        return False

    def evaluate(self, data_loader):
        model = self.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for (xi, y) in data_loader:
                xi = xi.long()
                if self.use_cuda:
                    xi, y = xi.cuda(), y.cuda()
                outputs = model(xi)
                y_pred.extend(outputs.cpu().data.numpy())
                y_true.extend(y.cpu().data.numpy())
        predictions_bounded = np.maximum(y_pred, np.ones(data_loader.dataset.__len__()) * min(y_true))
        predictions_bounded = np.minimum(predictions_bounded, np.ones(data_loader.dataset.__len__()) * max(y_true))
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        MAE = mean_absolute_error(y_true, predictions_bounded)
        if self.metric_type == 'MAE':
            return MAE
        else:
            return RMSE


if __name__ == '__main__':
    data = LoadData(pars.path, pars.dataset)

    save_path = ''
    if pars.save:
        save_path = '../save/' + pars.dataset + '/' + pars.metric + '/IBFM.pt' 

    model = IBFM(features_M=data.features_M, field_size=pars.field_size, embedding_size=pars.embedding_size,
                epochs=pars.epochs, batch_size=pars.batch_size, metric_type=pars.metric, l2=pars.l2, la=pars.la,
                learning_rate=pars.lr, att_dropout=pars.att_dropout, dnn_dropout=pars.dnn_dropout, dnnlayers=pars.dnnlayers, 
                attention_size=pars.attention_size, pretrain=pars.pretrain, save_path=save_path, use_cuda=True, verbose=pars.verbose)

    model = model.cuda()
    train_dataset = FrappeDataSet(data.Train_data)
    valid_dataset = FrappeDataSet(data.Validation_data)
    test_dataset = FrappeDataSet(data.Test_data)
    train_DataLoader = DataLoader(train_dataset, batch_size=pars.batch_size, shuffle=True, num_workers=1)
    valid_DataLoader = DataLoader(valid_dataset, batch_size=pars.batch_size, shuffle=False, num_workers=1)
    test_DataLoader = DataLoader(test_dataset, batch_size=pars.batch_size, shuffle=False, num_workers=1)
    t1 = time()
    model.fit(train_DataLoader, valid_DataLoader, test_DataLoader)

    best_valid_score = 0
    best_valid_score = min(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    print("Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f, [%.1f s]"
          % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], model.test_rmse[best_epoch],
             time() - t1))
