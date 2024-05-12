from trainer import ml_train, graph_train

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--m', default='lr', type=str)
    parser.add_argument('--bs', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--hidden_features', default=64, type=int)
    parser.add_argument('--epoch', default=10, type=int)


    args = parser.parse_args()

    if args.m in ['lr', 'lasso', 'ridge', 'bayridge', 'lor', 'svm', 'knn', 'gpr', 'dt', 'rf']:
        ml_train(args)
    elif args.m in ['graph']:
        graph_train(args)