from ds import get_data_numpy, get_data_tgeo
from models import get_model

from sklearn.metrics import mean_squared_error, median_absolute_error

from torch_geometric.loader import DataLoader

from torch.optim import Adam
from torch.nn import MSELoss, L1Loss

import numpy as np
import json
import hashlib
import os
import torch


def get_hash(args):
    args_str = json.dumps(vars(args), sort_keys=True)
    args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
    return args_hash

def save_json(dct, path):
    with open(path, 'w') as outfile:
        json.dump(dct, outfile)

def read_json(path):
    return json.load(open(path, 'r'))

def folder_setup(args):

    run_name = get_hash(args)

    run_dir = os.getcwd() + "/runs"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    save_dir = run_dir + f"/{run_name}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_json(vars(args), save_dir + f"/config.json")

    return save_dir


def ml_train(args):

    save_dir = folder_setup(args)

    Xs, Ts, scale = get_data_numpy()

    print(Xs.shape, Ts.shape)
    print(Xs.min(), Ts.min())
    print(Xs.max(), Ts.max())

    fold_cnt = Xs.shape[0]

    log = {}

    for fold_idx in range(fold_cnt):

        train_idx = list(range(fold_cnt))
        train_idx.remove(fold_idx)
        valid_idx = fold_idx


        X_train = np.concatenate(Xs[train_idx, :])
        T_train = np.concatenate(Ts[train_idx, :])

        X_valid = Xs[valid_idx, :]
        T_valid = Ts[valid_idx, :]

        model = get_model(args)

        model.fit(X_train, T_train)
        train_score = model.score(X_train, T_train)
        valid_score = model.score(X_valid, T_valid)

        print(f"Score - {fold_idx}", train_score, valid_score)

        T_tpred = model.predict(X_train)
        T_vpred = model.predict(X_valid)

        train_mse = mean_squared_error(T_tpred, T_train).item()
        valid_mse = mean_squared_error(T_vpred, T_valid).item()

        print(f"MSE - {fold_idx}", train_mse, valid_mse)

        train_mae = median_absolute_error(T_tpred, T_train).item()
        valid_mae = median_absolute_error(T_vpred, T_valid).item()

        print(f"MAE - {fold_idx}", train_mae, valid_mae)

        log[fold_idx] = {'train_mse' : train_mse, 'train_mae' : train_mae, 'valid_mse' : valid_mse, 'valid_mae' : valid_mae}

    save_json(log, save_dir + "/results.json")


def graph_train(args):

    save_dir = folder_setup(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = MSELoss()
    metric = L1Loss()

    folds, scale = get_data_tgeo()

    print(f"#Folds: {len(folds)}")

    fold_cnt = len(folds)

    log = {x : [] for x in range(fold_cnt)}

    for fold_idx in range(fold_cnt):

        valid_graphs = folds[fold_idx]
        train_graphs = sum([x for i, x in enumerate(folds) if i != fold_idx], [])

        train_ld = DataLoader(train_graphs, batch_size=args.bs, shuffle=True)
        valid_ld = DataLoader(valid_graphs, batch_size=args.bs, shuffle=True)

        print(f"Number Train Batch: {len(train_ld)}")
        print(f"Number Valid Batch: {len(valid_ld)}")

        model = get_model(args).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
    
        for epoch in range(args.epoch):
            print(f"Epoch: {epoch}")
            train_total_mse = 0
            train_total_mae = 0
            model.train()
            for data in train_ld:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out, data.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_total_mse += criterion(out, data.y).item() / len(train_ld)
                train_total_mae += metric(out, data.y).item() / len(train_ld)
        
            model.eval()
            with torch.no_grad():
                valid_total_mse = 0
                valid_total_mae = 0
                for data in valid_ld:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    loss = criterion(out, data.y)

                    valid_total_mse += criterion(out, data.y).item() / len(valid_ld)
                    valid_total_mae += metric(out, data.y).item() / len(valid_ld)
            
            print(f"\tMSE - {fold_idx}", train_total_mse, valid_total_mse)
            print(f"\tMAE - {fold_idx}", train_total_mae, valid_total_mae)

            log[fold_idx].append({'train_mse' : train_total_mse, 'train_mae' : train_total_mae, 'valid_mse' : valid_total_mse, 'valid_mae' : valid_total_mae})


    save_json(log, save_dir + "/results.json")