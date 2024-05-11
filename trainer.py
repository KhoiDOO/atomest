from ds import get_data_numpy, get_data_tgeo
from models import get_model

from sklearn.metrics import mean_squared_error

import numpy as np
import json
import hashlib
import os


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

    save_dir = os.getcwd() + "/runs"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_json(vars(args), save_dir + f"/config.json")

    return save_dir


def ml_train(args):

    save_dir = folder_setup(args)

    Xs, Ts = get_data_numpy()

    fold_cnt = Xs.shape[0]

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

        train_score = mean_squared_error(T_tpred, T_train)
        valid_score = mean_squared_error(T_vpred, T_valid)

        print(f"MSE - {fold_idx}", train_score, valid_score)