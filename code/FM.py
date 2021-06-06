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


parser = argparse.ArgumentParser(description='Hyperparameter tuning')
parser.add_argument('-use_cuda', default=1, type=int)
parser.add_argument('-path', default='../data/', type=str, help='Input data path')
parser.add_argument('-dataset', default='frappe', type=str, help='Choose a dataset.')
parser.add_argument('-pretrain', default=-1, type=int, help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save to pretrain file')
parser.add_argument('-epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('-metric', default='RMSE', type=str, help='MAE or RMSE')
parser.add_argument('-batch_size', default=2048, type=int, help='batch size')
parser.add_argument('-embedding_size', default=256, type=int, help='embedding size')
parser.add_argument('-field_size', default=10, type=int, help='field size of dataset')
parser.add_argument('-l2', default=0, type=float, help='Regularizer')
parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
parser.add_argument('-save', default=1, type=int, help='save model state_dict')
parser.add_argument('-dropout', default=0.3, type=float, help='dropout ratio')
parser.add_argument('-verbose', default=1, type=int, help='Whether to show the performance of each epoch (0 or 1)')
pars = parser.parse_args()

if not torch.cuda.is_available():
    pars.use_cuda = 0

class FM(nn.Module):
    def __init__(self, features_M, field_size, embedding_size, epochs, metric_type, learning_rate, l2,
                dropout, pretrain, save_path, verbose, use_cuda, random_seed=2021):
        super(FM, self).__init__()
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.dropout = dropout
        self.metric_type = metric_type
        self.verbose = verbose
        self.random_seed = random_seed
        self.features_M = features_M
        self.pretrain = pretrain
        self.use_cuda = use_cuda
        self.save_path = save_path
        self.num_pair = self.field_size * (self.field_size - 1) // 2

        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)

        self.bias = nn.Parameter(torch.tensor([.0]))
        self.fm_1st_embedding = nn.Embedding(self.features_M, 1)
        self.fm_2nd_embedding = nn.Embedding(self.features_M, self.embedding_size)

        nn.init.uniform_(self.fm_1st_embedding.weight.data, 0.0, 0.0)
        nn.init.normal_(self.fm_2nd_embedding.weight.data, 0, 0.01)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, xi):
        fm_1st_embedding = self.fm_1st_embedding(xi).squeeze()
        fm_2nd_embedding = self.fm_2nd_embedding(xi)

        interaction_part1 = torch.pow(torch.sum(fm_2nd_embedding, 1), 2)
        interaction_part2 = torch.sum(torch.pow(fm_2nd_embedding, 2), 1)

        fm_2nd_order = 0.5 * torch.sub(interaction_part1, interaction_part2)
        fm_2nd_order = self.dropout_layer(fm_2nd_order)
        
        total_sum = torch.sum(fm_1st_embedding, 1) + torch.sum(fm_2nd_order, 1) + self.bias
        return total_sum

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
                y = y.float()
                if self.use_cuda:
                    xi, y = xi.cuda(), y.cuda()
                optimizer.zero_grad()
                y_pred = model(xi)
                loss = criterion(y, y_pred)
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

            if self.eva_termination(self.valid_rmse):
                break
            
        if self.pretrain == -1:
            bias = model.bias.cpu().detach().numpy()
            fm_1st_embedding = model.fm_1st_embedding.cpu().weight.data.numpy()
            fm_2nd_embedding = model.fm_2nd_embedding.cpu().weight.data.numpy()
            np.save('../pretrain/bias.npy', bias)
            np.save('../pretrain/fm_1st.npy', fm_1st_embedding)
            np.save('../pretrain/fm_2nd.npy', fm_2nd_embedding)

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
                y = y.float()
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
            save_path = '../save/' + pars.dataset + '/' + pars.metric + '/FM.pt' 
        model = FM(features_M=data.features_M, field_size=pars.field_size, embedding_size=pars.embedding_size,
                      epochs=pars.epochs, metric_type=pars.metric, l2=pars.l2, learning_rate=pars.lr, dropout=pars.dropout, 
                      pretrain=pars.pretrain, save_path=save_path, use_cuda=pars.use_cuda, verbose=pars.verbose)
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
          % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], model.test_rmse[best_epoch], time() - t1))