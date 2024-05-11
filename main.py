from trainer import ml_train

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--m', default='lr', type=str)

    args = parser.parse_args()

    ml_train(args)