from scipy import io
from torch_geometric.data import Data
from tqdm import tqdm

import numpy as np
import torch
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt

STEP = 1
NOISE = 1

def realize(X, triuind):
    def _realize_(x):
        inds = np.argsort(-(x**2).sum(axis=0)**.5+np.random.normal(0,NOISE,x[0].shape))
        x = x[inds,:][:,inds]*1
        x = x.flatten()[triuind]
        return x
    return np.array([_realize_(z) for z in X])

def expand(X, _max):
    Xexp = []
    for i in range(X.shape[1]):
        for k in np.arange(0,_max[i]+STEP,STEP):
            Xexp += [np.tanh((X[:,i]-k)/STEP)]
    return np.array(Xexp).T

def save_pickle(dct, path):
    with open(path, 'wb') as  file:
        pickle.dump(obj=dct, file=file, protocol=pickle.HIGHEST_PROTOCOL)
    
def read_pickle(path):
    with open(path, 'rb') as  file:
        dct = pickle.load(file=file)
    file.close()
    return dct

def make_graphs(X_fold, T_fold, R_fold):
    lst = []

    for x, t, r in tqdm(zip(X_fold, T_fold, R_fold)):
        # adj_mat = torch.masked_fill(x != 0, torch.eye(23, 23, dtype=torch.bool), 0)

        mask = x != 0 
        mask_flat = mask.flatten()

        edge_index = torch.argwhere(x != 0).T

        edge_attr = x.flatten()[mask_flat].unsqueeze(-1)

        data = Data(x=r, edge_index=edge_index, edge_attr=edge_attr, y=t)

        lst.append(data)
    
    return lst


def get_data_numpy():
    data_path = "/".join(__file__.split("/")[:-1]) + f"/src/qm7.mat"
    dataset = io.loadmat(data_path)

    X = dataset['X']
    R = dataset['R']
    Z = dataset['Z']
    T = dataset['T']
    P = dataset['P']

    triuind = (np.arange(23)[:,np.newaxis] <= np.arange(23)[np.newaxis,:]).flatten()
    _max = 0
    for _ in range(10): 
        _max = np.maximum(_max, realize(X, triuind).max(axis=0))
    
    X = realize(X, triuind)
    mean = X.mean(axis=0)
    std = X.std()

    X_norm = (X-mean)/std

    T_norm = T.flatten() / T.max()

    Xs, Ts = X_norm[P], T_norm[P]

    return Xs, Ts

def get_data_tgeo():

    save_path = "/".join(__file__.split("/")[:-1]) + "/src/data.pickle"
    
    if not os.path.exists(save_path):
    
        data_path = "/".join(__file__.split("/")[:-1]) + f"/src/qm7.mat"
        dataset = io.loadmat(data_path)

        X = dataset['X']
        R = dataset['R']
        Z = dataset['Z']
        T = dataset['T']
        P = dataset['P']

        X = torch.from_numpy(X)
        T = torch.from_numpy(T)
        R = torch.from_numpy(R)

        T = T.flatten() / T.min().abs()

        X_max = X.max(dim=0).values

        X = X / X_max

        Xs = X[P]
        Ts = T[P]
        Rs = R[P]

        folds = []

        for fold_idx in range(Xs.shape[0]):
            graphs = make_graphs(Xs[fold_idx], Ts[fold_idx], Rs[fold_idx])

            folds.append(graphs)
        
        save_pickle({'data' : folds}, save_path)

        return folds
    
    else:
        folds = read_pickle(save_path)['data']

        return folds


if __name__ == "__main__":
    Xs, Ts = get_data_numpy()

    print(Xs.shape)
    print(Ts.shape)

    folds = get_data_tgeo()

    print(len(folds))

    sample_fold = folds[0]

    print(len(sample_fold))

    sample_data = sample_fold[0]

    print(sample_data)