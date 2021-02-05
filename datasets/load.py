import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import os
import heapq
import sys
import_path = os.path.split(os.path.realpath(__file__))[0] + '/..'
if import_path not in sys.path:
    sys.path.append(import_path)
import time
import random
from itertools import product
from sklearn.preprocessing import MinMaxScaler

from utils.train import Trainer, Plugin
from metrics.RootRelativeSquaredError import RRSE
from metrics.EmpiricalCorrelaionCoefficient import CORR
from metrics.MeanAbsoluteError import MAE
from metrics.MeanSquaredError import MSE
from itertools import product

def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def grid_search(*hyper_params):
    for hyper_param in product(*hyper_params):
        yield hyper_param

def get_name_from_params(params):
    params = [str(p) for p in params]
    f = '_'.join(params)
    return f

def get_with_pickle(fn):
    with open(fn, 'rb') as f:
        result = pickle.load(f)
    return result

def save_with_pickle(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)

class MTSDataset(torch.utils.data.Dataset):
    # data :numpy.array 
    # data.shape = (N, feature)
    #mode in ['train', 'valid', 'test']
    def __init__(self, data, window, horizon, mm, mode='train'):
        N, _ = data.shape
        self.window = window
        self.horizon = horizon
        self.mode = mode

        self.nums = N - window - horizon + 1
        self.train_nums = int(self.nums*0.6)
        self.valid_nums = int(self.nums*0.2)

        self.data = torch.from_numpy(mm.fit_transform(data)).float()
        self.mm = mm
    
    def set_mode(self, mode):
        self.mode = mode


    def __getitem__(self, index):
        begin = None
        if self.mode == 'train':    
            begin = 0
        elif self.mode == 'valid':
            begin = self.train_nums
        elif self.mode == 'test':
            begin = self.train_nums + self.valid_nums

        x = self.data[begin+index:begin+index+self.window, :]
        y = self.data[begin+index+self.window+self.horizon-1, :]
        return x, y

    def __len__(self):
        if self.mode == 'train':
            return self.train_nums
        elif self.mode == 'valid':
            return self.valid_nums
        elif self.mode == 'test':
            return self.nums - self.train_nums - self.valid_nums

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_electricity():
    current_file_name = os.path.realpath(__file__)
    current_file_path = os.path.split(current_file_name)[0]
    electricity_file_name = os.path.join(current_file_path, 'electricity/electricity.npy')
    return np.load(electricity_file_name)

def get_exchange_rate():
    current_file_name = os.path.realpath(__file__)
    current_file_path = os.path.split(current_file_name)[0]
    exchange_rate_file_name = os.path.join(current_file_path, 'exchange_rate/exchange_rate.npy')
    return np.load(exchange_rate_file_name)

def get_solar_energy():
    current_file_name = os.path.realpath(__file__)
    current_file_path = os.path.split(current_file_name)[0]
    solar_energy_file_name = os.path.join(current_file_path, 'solar-energy/solar_AL.npy')
    return np.load(solar_energy_file_name)

def get_traffic():
    current_file_name = os.path.realpath(__file__)
    current_file_path = os.path.split(current_file_name)[0]
    traffic_file_name1 = os.path.join(current_file_path, 'traffic/traffic1.npy')
    traffic_file_name2 = os.path.join(current_file_path, 'traffic/traffic2.npy')
    traffic1 =  np.load(traffic_file_name1)
    traffic2 = np.load(traffic_file_name2)
    traffic = np.concatenate([traffic1, traffic2], 0)
    return traffic

def get_dataset(dataset_name, window, horizon):
    dataset_dict = {
        'solar_energy':get_solar_energy,
        'traffic':get_traffic,
        'electricity':get_electricity,
        'exchange_rate':get_exchange_rate
    }
    get = dataset_dict[dataset_name]
    data = get()
    dataset = MTSDataset(data, window, horizon, MinMaxScaler())
    return dataset

class MTSTrainer(Trainer):
    def __init__(self, model, criterion, optimizer, dataset):
        super(MTSTrainer, self).__init__(model, criterion, optimizer, dataset)
        self.jump_out = False
        self.original_dataset = dataset
        self.dataset = None
    
    def run(self, epochs=1, batch_size=64, shuffle=True):
        for q in self.plugin_queues.values():
            heapq.heapify(q)
        self.dataset = torch.utils.data.DataLoader(self.original_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
        for i in range(1, epochs + 1):
            self.train(batch_size, shuffle)
            self.call_plugins('epoch', i, i == epochs)
            if self.jump_out:
                break
        self.dataset = None

    def train(self, batch_size, shuffle=True):
        self.original_dataset.set_mode('train')
        super(MTSTrainer, self).train()

class MTSPlugin(Plugin):
    def __init__(self, file_path, limit=10):
        super(MTSPlugin, self).__init__([(1, 'epoch'), (1, 'iteration')])
        self.file_path = file_path
        self.iteration_nums = 0
        self.iteration_loss = 0
        self.limit = limit
        self.counter = 0
        self.validation_best_corr = -1
        self.validation_best_rse = 1
        self.mm = None
        #self.last_time = time.time()
    
    def register(self, trainer):
        self.trainer = trainer
        self.mm = trainer.original_dataset.mm
        self.trainer.stats['valid_RSE'] = []
        self.trainer.stats['valid_CORR'] = []
        self.trainer.stats['valid_MSE'] = []
        self.trainer.stats['valid_MAE'] = []

        self.trainer.stats['test_RSE'] = []
        self.trainer.stats['test_CORR'] = []
        self.trainer.stats['test_MSE'] = []
        self.trainer.stats['test_MAE'] = []

        self.trainer.stats['train_Loss'] = []
        
    def epoch(self, time, last_time):
        self.trainer.model.eval()
        stats = self.trainer.stats

        train_loss = self.iteration_loss/self.iteration_nums
        stats['train_Loss'].append(train_loss)
        self.iteration_nums = 0
        self.iteration_loss = 0

        valid_rrse = 0.0
        valid_corr = 0.0
        valid_mse = 0.0
        valid_mae = 0.0

        test_rrse = 0.0
        test_corr = 0.0
        test_mse = 0.0
        test_mae = 0.0

        #mse = nn.MSELoss()
        with torch.no_grad():
            self.trainer.original_dataset.set_mode('valid')
            dataloader = torch.utils.data.DataLoader(self.trainer.original_dataset, batch_size=64, shuffle=False, pin_memory=True)
            valid_pred = []
            valid_y = []
            for (X, Y) in dataloader:
                X = X.cuda()
                predY = self.trainer.model(X)
                valid_pred.append(predY)
                valid_y.append(Y)
                
            Y = torch.cat(valid_y, 0)
            predY = torch.cat(valid_pred, 0)

            predY = predY.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
            predY = self.mm.inverse_transform(predY)
            Y = self.mm.inverse_transform(Y)
            
            rrse, corr = RRSE(Y, predY), CORR(Y, predY)
            valid_corr = float(corr)
            valid_rrse= float(rrse)
            valid_mse = float(MSE(Y, predY))
            valid_mae = float(MAE(Y, predY))

            self.trainer.original_dataset.set_mode('test')
            test_pred = []
            test_y = []
            for (X, Y) in dataloader:
                X = X.cuda()
                #Y = Y.cuda()
                predY = self.trainer.model(X)
                test_pred.append(predY)
                test_y.append(Y)

                
            Y = torch.cat(test_y, 0)
            predY = torch.cat(test_pred, 0)

            predY = predY.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
            predY = self.mm.inverse_transform(predY)
            Y = self.mm.inverse_transform(Y)
    
            rrse, corr = RRSE(Y, predY), CORR(Y, predY)
            test_corr = float(corr)
            test_rrse = float(rrse)
            test_mse = float(MSE(Y, predY))
            test_mae = float(MAE(Y, predY))

        stats['valid_RSE'].append(valid_rrse)
        stats['valid_CORR'].append(valid_corr)
        stats['valid_MSE'].append(valid_mse)
        stats['valid_MAE'].append(valid_mae)
        stats['test_RSE'].append(test_rrse)
        stats['test_CORR'].append(test_corr)
        stats['test_MSE'].append(test_mse)
        stats['test_MAE'].append(test_mae)


        self.counter += 1
        if valid_corr - valid_rrse >= self.validation_best_corr - self.validation_best_rse:
            self.validation_best_corr = valid_corr
            self.validation_best_rse = valid_rrse
            self.counter = 0
            self.save_model()
        
        if self.counter >= self.limit:#提前结束需要保存数据
            self.trainer.jump_out = True
            self.save_stats()
        elif last_time:#如果是最后一次运行也需要保存数据
            self.save_stats()
        self.trainer.model.train()

        #print(f'epochs={time},train_mse={mse},valid_corr={valid_mean_corr},valid_rse={valid_mean_rrse},test_corr={test_mean_corr},test_rse={test_mean_rrse}')
        print(f'epochs={time}')
        print(f'train:loss={train_loss}')
        print(f'valid:mse={valid_mse},mae={valid_mae},rse={valid_rrse},corr={valid_corr}')
        print(f'test:mse={test_mse},mae={test_mae},rse={test_rrse},corr={test_corr}')

    
    def iteration(self, time, batch_input, batch_target, batch_output, loss):
        L = loss.item()
        #c = time.time()
        #print(f'{c-self.last_time}')
        #self.last_time = c
        self.iteration_loss += L
        self.iteration_nums += 1

    def save_model(self):
        model_file = self.file_path + '.model'
        #with open(model_file, 'wb') as f:
        #    pickle.dump(self.trainer.model.cpu(), f)
        #self.trainer.model.cuda()
        torch.save(self.trainer.model.state_dict(), model_file)

    # def load_model(self):
    #     model_file = self.file_path + '.model'
    #     return torch.load(model_file)
    
    def save_stats(self):
        stats_file = self.file_path + '.stats'
        with open(stats_file, 'wb') as f:
            pickle.dump(self.trainer.stats, f)

    # def load_stats(self):
    #     stats_file = self.file_path + '.stats'
    #     result = None
    #     with open(stats_file, 'rb') as f:
    #         result = pickle.load(f)
    #     return result


if __name__ == '__main__':
    electricity = get_electricity()
    print('electricity', electricity.shape)
    exchange_rate = get_exchange_rate()
    print('exchange_rate', exchange_rate.shape)
    solar_energy = get_solar_energy()
    print('solar_energy', solar_energy.shape)
    traffic = get_traffic()
    print('traffic', traffic.shape)