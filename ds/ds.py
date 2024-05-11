from scipy import io
import numpy as np

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
    
    X = expand(realize(X, triuind),_max)
    mean = X.mean(axis=0)
    std = (X - mean).std()

    X_norm = (X-mean)/std

    T_norm = T.flatten() / 2000.

    Xs, Ts = X_norm[P], T_norm[P]

    return Xs, Ts

def get_data_tgeo():
    pass


if __name__ == "__main__":
    Xs, Ts = get_data_numpy()

    print(Xs.shape)
    print(Ts.shape)