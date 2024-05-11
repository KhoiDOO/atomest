from scipy import io
import numpy as np


if __name__ == "__main__":
    data_path = "/".join(__file__.split("/")[:-1]) + f"/src/qm7.mat"

    dataset = io.loadmat(data_path)

    print(type(dataset))
    print(dataset.keys())

    x = dataset['X']
    r = dataset['R']
    z = dataset['Z']
    t = dataset['T']
    p = dataset['P']

    """
    The dataset is composed of three multidimensional arrays X (7165 x 23 x 23), T (7165) and P (5 x 1433) representing the inputs (Coulomb matrices), 
    the labels (atomization energies) and the splits for cross-validation, respectively. The dataset also contain two additional multidimensional arrays 
    Z (7165) and R (7165 x 3) representing the atomic charge and the cartesian coordinate of each atom in the molecules.
    """

    print(f"x shape: {x.shape}")
    print(f"r shape: {r.shape}")
    print(f"z shape: {z.shape}")
    print(f"t shape: {t.shape}")
    print(f"p shape: {p.shape}")

    # print(f"x samples:\n{x[0]}")
    # print(f"r samples:\n{r[0]}")
    # print(f"z samples:\n{z[0]}")
    # print(f"t samples:\n{t[:, 0]}")
    # print(f"p samples:\n{p[0]}")

    print(f"check positive t: {np.sum(t[0] <= 0)}")
    print(f"t min: {t.min()}")
    print(f"t mean: {t.mean()}")