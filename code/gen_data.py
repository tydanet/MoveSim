import numba as nb
import numpy as np

from utils import read_data_from_file, read_gps


def gen_matrix(dataset="geolife"):
    train_data = read_data_from_file(f"../data/{dataset}/real.data")
    gps = read_gps(f"../data/{dataset}/gps")

    if dataset == "mobile":
        max_locs = 8606
    else:
        max_locs = 23768

    reg1, reg2 = gen_matrix_inner(train_data, max_locs, gps)

    np.save(f"../data/{dataset}/M1.npy", reg1)
    np.save(f"../data/{dataset}/M2.npy", reg2)

    print("Matrix Generation Finished")


@nb.njit(parallel=True)
def gen_matrix_inner(train_data, max_locs, gps):
    regs = np.zeros((2, max_locs, max_locs))

    reg1 = regs[0]  # weighted adjacency matrix
    reg2 = regs[1]  # distance matrix

    for row in train_data:
        for k in range(len(row) - 1):
            i = row[k]
            j = row[k + 1]
            reg1[i, j] += 1

    for i in nb.prange(reg2.shape[0]):
        for j in range(reg2.shape[1]):
            if i >= j:
                reg2[i, j] = np.linalg.norm(gps[i] - gps[j])

    reg2 = reg2 + reg2.T
    return regs
