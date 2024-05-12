from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, LogisticRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool

import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(GCN, self).__init__()
        self.conv1 = GATConv(in_features, hidden_features, edge_dim=1)
        self.conv2 = GATConv(hidden_features, hidden_features, edge_dim=1)
        self.conv3 = GATConv(hidden_features, hidden_features, edge_dim=1)
        self.lin = Linear(hidden_features, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x.squeeze(-1)


def get_model(args):
    if args.m == 'lr':
        return LinearRegression()
    elif args.m == 'lasso':
        return Lasso()
    elif args.m == 'ridge':
        return Ridge()
    elif args.m == 'bayridge':
        return BayesianRidge()
    elif args.m == 'lor':
        return LogisticRegression()
    elif args.m == 'svm':
        return SVR()
    elif args.m == 'knn':
        return KNeighborsRegressor()
    elif args.m == 'gpr':
        return GaussianProcessRegressor()
    elif args.m == 'dt':
        return DecisionTreeRegressor()
    elif args.m == 'rf':
        return RandomForestRegressor()
    elif args.m == 'graph':
        return GCN(in_features=3, hidden_features=args.hidden_features)
    else:
        raise ValueError(f'not support model {args.m}')