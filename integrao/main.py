import numpy as np
import pandas as pd
import time
from snf.compute import _flatten, _B0_normalized, _find_dominate_set
from sklearn.utils.validation import (
    check_array,
    check_symmetric,
    check_consistent_length,
)

def dist2(X, C):
    """
    Description: Computes the Euclidean distances between all pairs of data point given

    Usage: dist2(X, C)
    X: A data matrix where each row is a different data point
    C: A data matrix where each row is a different data point. If this matrix is the same as X,
    pairwise distances for all data points in X are computed.

    Return: Returns an N x M matrix where N is the number of rows in X and M is the number of rows in C.

    Author: Dr. Anna Goldenberg, Bo Wang, Aziz Mezlini, Feyyaz Demir
    Python Version Rewrite: Rex Ma

    Examples:
        # Data1 is of size n x d_1, where n is the number of patients, d_1 is the number of genes,
        # Data2 is of size n x d_2, where n is the number of patients, d_2 is the number of methylation
        Dist1 = dist2(Data1, Data1)
        Dist2 = dist2(Data2, Data2)
    """

    ndata = X.shape[0]
    ncentres = C.shape[0]

    sumsqX = np.sum(X * X, axis=1)
    sumsqC = np.sum(C * C, axis=1)

    XC = 2 * (np.matmul(X, np.transpose(C)))

    res = (
        np.transpose(np.reshape(np.tile(sumsqX, ncentres), (ncentres, ndata)))
        + np.reshape(np.tile(sumsqC, ndata), (ndata, ncentres))
        - XC
    )

    return res


def _find_dominate_set_relative(W, K=20):
    """
    Retains `K` strongest edges for each sample in `W`
    Parameters
    ----------
    W : (N, N) array_like
        Input data
    K : (0, N) int, optional
        Number of neighbors to retain. Default: 20
    Returns
    -------
    Wk : (N, N) np.ndarray
        Thresholded version of `W`
    """

    # let's not modify W in place
    Wk = W.copy()

    # determine percentile cutoff that will keep only `K` edges for each sample
    # remove everything below this cutoff
    cutoff = 100 - (100 * (K / len(W)))
    Wk[Wk < np.percentile(Wk, cutoff, axis=1, keepdims=True)] = 0

    # normalize by strength of remaining edges
    Wk = Wk / np.nansum(Wk, axis=1, keepdims=True)

    Ws = Wk + np.transpose(Wk)

    return Ws


def _stable_normalized(W):
    """
    Adds `alpha` to the diagonal of `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF

    Returns
    -------
    W : (N, N) np.ndarray
        Stable-normalized similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    rowSum = np.sum(W, 1) - np.diag(W)
    rowSum[rowSum == 0] = 1

    W = W / (2 * rowSum)
    np.fill_diagonal(W, 0.5)
    W = check_symmetric(W, raise_warning=False)

    return W


def _stable_normalized_pd(W):
    """
    Adds `alpha` to the diagonal of pandas dataframe `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF

    Returns
    -------
    W : (N, N) np.ndarray
        Stable-normalized similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    rowSum = np.sum(W, 1) - np.diag(W)
    rowSum[rowSum == 0] = 1

    W = W / (2 * rowSum)

    W_np = W.values
    np.fill_diagonal(W_np, 0.5)
    W = pd.DataFrame(W_np, index=W.index, columns=W.columns) 

    W = check_symmetric(W, raise_warning=False)

    return W


def _scaling_normalized_pd(W, ratio):
    """
    Adds `alpha` to the diagonal of pandas dataframe `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF

    Returns
    -------
    W : (N, N) np.ndarray
        Stable-normalized similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    rowSum = np.sum(W, 1) - np.diag(W)
    rowSum[rowSum == 0] = 1

    W = (W / rowSum) * 0.5 * ratio 

    W_np = W.values
    np.fill_diagonal(W_np, 1-0.5*ratio)
    W = pd.DataFrame(W_np, index=W.index, columns=W.columns) 

    W = check_symmetric(W, raise_warning=False)

    return W


def handle_zeros_in_scale(scale, copy=True):
    """Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    Adapted from sklearn.preprocessing.data"""

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
    return scale


def integrao_fuse(aff, dicts_common, dicts_unique, original_order, neighbor_size=20, fusing_iteration=20,
                  normalization_factor=1.0):
    """
    Performs Patient Graph Fusion on `aff` matrices

    Parameters
    ----------
    aff : (N, N) pandas dataframe
        Input similarity arrays; all arrays should be square but no need to be equal size.

    dicts_common: dictionaries, required
        Dictionaries for getting common samples from different views
        Example: dicts_common[(0, 1)] == dicts_common[(1, 0)], meaning the common patients between view 1&2

    dicts_unique: dictionaries, required
        Dictionaries for getting unique samples for different views
        Example: dicts_unique[(0, 1)], meaning the unique samples from view 1 that are not in view 2
                 dicts_unique[(1, 0)], meaning the unique samples from view 2 that are not in view 1

    original_order: lists, required 
        The original order of each view

    K : (0, N) int, optional
        Hyperparameter normalization factor for scaling. Default: 20

    t : int, optional
        Number of iterations to perform information swapping. Default: 20

    alpha : (0, 1) float, optional
        Hyperparameter normalization factor for scaling. Default: 1.0

    Returns
    -------
    W: (N, N) Ouputs similarity arrays
        Fused similarity networks of input arrays
    """

    print("Start applying diffusion!")
    start_time = time.time()
    newW = [0] * len(aff)

    for n, mat in enumerate(aff):
        t0 = time.time()
        aff[n] = _stable_normalized_pd(mat)
        neighbor_size = min(int(neighbor_size), mat.shape[0])
        newW[n] = _find_dominate_set(aff[n], neighbor_size)
        print(f"[{n}] normalization_and_knn: {time.time() - t0:.4f}s")

    if len(aff) == 1:
        print("Only one view, return it directly")
        return newW

    for iteration in range(fusing_iteration):
        print(f"\n-- Iteration {iteration + 1} --")
        aff_next = []
        for k in range(len(aff)):
            t0 = time.time()
            aff_temp = aff[k].copy()
            for col in aff_temp.columns:
                aff_temp[col].values[:] = 0
            aff_next.append(aff_temp)
            print(f"View {k}: prepare_aff_next: {time.time() - t0:.4f}s")

        for n, mat in enumerate(aff):
            nzW = newW[n]
            for j, mat_tofuse in enumerate(aff):
                if n == j:
                    continue

                t0 = time.time()
                sample_order = np.array(sorted(dicts_common[(j, n)]) + sorted(dicts_unique[(j, n)]))
                mat_tofuse = mat_tofuse.loc[sample_order, sample_order]

                num_common = len(dicts_common[(n, j)])
                mat_cols = mat_tofuse.columns
                mat_shape = mat_tofuse.shape
                to_drop_mat = mat_cols[num_common: mat_shape[1]].values.tolist()
                mat_tofuse_crop = mat_tofuse.drop(to_drop_mat, axis=1).drop(to_drop_mat, axis=0)
                print(f"[{n},{j}] reindex_and_crop: {time.time() - t0:.4f}s")




                t0 = time.time()
                # nzW_identity = pd.DataFrame(np.identity(nzW.shape[0]), index=original_order[n],
                #                             columns=original_order[n])
                # mat_tofuse_union = nzW_identity + mat_tofuse_crop
                # mat_tofuse_union.fillna(0.0, inplace=True)
                # mat_tofuse_union = _scaling_normalized_pd(mat_tofuse_union,
                #                                           ratio=mat_tofuse_crop.shape[0] / nzW_identity.shape[0])

                mat_tofuse_union = mat_tofuse_crop.values.copy()
                np.fill_diagonal(mat_tofuse_union, mat_tofuse_union.diagonal() + 1.0)

                rowSum = np.sum(mat_tofuse_union, axis=1) - np.diag(mat_tofuse_union)
                rowSum[rowSum == 0] = 1
                mat_tofuse_union = (mat_tofuse_union.T / rowSum).T * 0.5 * (mat_tofuse_crop.shape[0] / nzW.shape[0])
                np.fill_diagonal(mat_tofuse_union, 1 - 0.5 * (mat_tofuse_crop.shape[0] / nzW.shape[0]))


                mat_tofuse_union = check_symmetric(mat_tofuse_union)
                mat_tofuse_union = mat_tofuse_union.reindex(original_order[n], axis=1).reindex(original_order[n],
                                                                                               axis=0)
                print(f"[{n},{j}] identity_addition: {time.time() - t0:.4f}s")


                if newW[n].shape[0] < 2000:
                    t0 = time.time()
                    nzW_T = np.transpose(nzW)
                    aff0_temp = nzW.dot(mat_tofuse_union.dot(nzW_T))
                    print(f"[{n},{j}] Dense multiplication: {time.time() - t0:.4f}s")
                else:
                    t0 = time.time()
                    nzW_sparse = csr_matrix(newW[n].values)
                    nzW_T_sparse = nzW_sparse.transpose()
                    mat_union_dense = mat_tofuse_union.values
                    intermediate = nzW_sparse @ mat_union_dense
                    aff0_temp_np = intermediate @ nzW_T_sparse
                    aff0_temp = pd.DataFrame(aff0_temp_np, index=original_order[n], columns=original_order[n])
                    print(f"[{n},{j}] Sparse multiplication: {time.time() - t0:.4f}s")

                t0 = time.time()
                aff0_temp = _stable_normalized_pd(aff0_temp)
                aff_next[n] = np.add(aff0_temp, aff_next[n])
                print(f"[{n},{j}] final_normalization: {time.time() - t0:.4f}s")

            aff_next[n] = np.divide(aff_next[n], len(aff) - 1)

        for k in range(len(aff)):
            aff[k] = aff_next[k]

    for n, mat in enumerate(aff):
        aff[n] = _stable_normalized_pd(mat)

    print("Diffusion ends! Times: {:.2f}s".format(time.time() - start_time))
    return aff