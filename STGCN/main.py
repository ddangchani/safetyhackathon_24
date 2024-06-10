import argparse
import logging
import math
import os
import pickle
import random
import time
import logging
from tqdm import tqdm

os.chdir('STGCN')

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from model.models import STGCNChebGraphConv, STGCNGraphConv
from utils import calc_gso, calc_chebynet_gso, evaluate_metric, evaluate_model, EarlyStopping, set_env, data_transform

# Load data
adj_mx = pd.read_csv('data/PeMSD7_W_228.csv', header=None)
adj_mx = np.array(adj_mx)
adj_mx = torch.from_numpy(adj_mx)

data = pd.read_csv('data/PeMSD7_V_228.csv', header=None)
data = np.array(data)
data = torch.from_numpy(data)

# Arguments

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--n_his', type=int, default=12, help='number of time intervals in the past')
    parser.add_argument('--n_pred', type=int, default=3, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5, help='time interval, default as 5')
    parser.add_argument('--Kt', type=int, default=3, help='kernel size of temporal convolution')
    parser.add_argument('--stblock_num', type=int, default=2, help='number of ST-Conv blocks')
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2], help='kernel size of spatial convolution')
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num #

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    return args, device, blocks

args, device, blocks = get_parameters()

# Data Preprocessing

adj, n_vertex = adj_mx, adj_mx.shape[0]
gso = calc_gso(adj, args.gso_type)

if args.graph_conv_type == 'cheb_graph_conv':
    gso = calc_chebynet_gso(gso)

gso = gso.toarray()
gso = gso.astype(dtype=np.float32)
args.gso = torch.from_numpy(gso).to(device)

data_col = data.shape[0]
val_and_test_rate = 0.15
len_val = int(math.floor(data_col * val_and_test_rate))
len_test = int(math.floor(data_col * val_and_test_rate))
len_train = int(data_col - len_val - len_test)

train, val, test = data[:len_train], data[len_train: len_train + len_val], data[-len_test:]
scaler = preprocessing.StandardScaler() # Standardize features by removing the mean and scaling to unit variance
train = scaler.fit_transform(train)
val = scaler.fit_transform(val)
test = scaler.fit_transform(test)

x_train, y_train = data_transform(train, args.n_his, args.n_pred, device)
x_val, y_val = data_transform(val, args.n_his, args.n_pred, device)
x_test, y_test = data_transform(test, args.n_his, args.n_pred, device)

train_data = TensorDataset(x_train, y_train)
train_iter = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
val_data = TensorDataset(x_val, y_val)
val_iter = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
test_data = TensorDataset(x_test, y_test)
test_iter = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

# Model

loss = nn.MSELoss()
early_stopping = EarlyStopping(patience=args.patience)

if args.graph_conv_type == 'cheb_graph_conv':
    model = STGCNChebGraphConv(args, blocks, n_vertex).to(device)
else:
    model = STGCNGraphConv(args, blocks, n_vertex).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# Training

def train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter):
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm(train_iter):
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        if es.step(val_loss):
            print('Early stopping.')
            break

@torch.no_grad()
def val(model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

@torch.no_grad() 
def test(zscore, loss, model, test_iter, args):
    model.eval()
    test_MSE = evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = evaluate_metric(model, test_iter, zscore)
    print(f'Dataset test MSE: {test_MSE:.6f}, MAE: {test_MAE:.6f}, RMSE: {test_RMSE:.6f}, WMAPE: {test_WMAPE:.6f}')

# Print input data shape
print(f'x_train.shape: {x_train.shape}')


# if __name__ == '__main__':

#     # Log
#     logging.basicConfig(level=logging.INFO)

#     # Training
#     train(loss, args, optimizer, scheduler, early_stopping, model, train_iter, val_iter)

#     # Testing
#     test(scaler, loss, model, test_iter, args)

# # Save model
# torch.save(model.state_dict(), 'model.pth')

# Load model
model = STGCNChebGraphConv(args, blocks, n_vertex).to(device)

model.load_state_dict(torch.load('model.pth'))

# Testing : Given the first 12 time intervals, predict the next 3 time intervals
first = data[:16]
first_12_time_intervals = scaler.fit_transform(first)
x_test, y_test = data_transform(first_12_time_intervals, args.n_his, args.n_pred, device) 
y_pred = model(x_test).view(len(x_test), -1)
y_pred = y_pred.cpu().detach().numpy()

# Plot
plt.plot(first_12_time_intervals, label='First 12 time intervals')
plt.plot(y_pred, label='Predicted next 3 time intervals')
plt.legend()
plt.show()