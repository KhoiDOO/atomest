from sklearn.linear_model import LinearRegression

def get_model(args):
    if args.m == 'lr':
        return LinearRegression()
    else:
        raise ValueError(f'not support model {args.m}')