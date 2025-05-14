import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.utils.validation import (
    check_symmetric,
)
from snf.compute import _find_dominate_set
from scipy.sparse import issparse
import faiss
import contexttimer


def dist2(X, C):
    """
    Computes pairwise squared Euclidean distances for a matrix X using faiss library.

    Parameters
    ----------
    X, C : np.ndarray or sparse
        Input data matrices.

    Returns
    -------
    D : ndarray
        Matrix of squared Euclidean distances
    """

    X = X.astype('float32')
    n = X.shape[0]
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    distances, indices = index.search(X, n)
    D_full = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n):
        D_full[i, indices[i]] = distances[i]  # squared distances
    np.fill_diagonal(D_full, 0.0)

    return D_full


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

    return Wk + np.transpose(Wk)


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
    np.fill_diagonal(W_np, 1 - 0.5 * ratio)
    W = pd.DataFrame(W_np, index=W.index, columns=W.columns)

    # W = check_symmetric(W, raise_warning=False)

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

    neighbor_size : (0, N) int, optional
        Hyperparameter neighbor_size. Default: 20

    fusing_iteration : int, optional
        Number of iterations to perform. Default: 20

    normalization_factor : (0, 1) float, optional
        Hyperparameter normalization factor. Default: 1.0

    Returns
    -------
    W: (N, N) Ouputs similarity arrays
        Fused similarity networks of input arrays
    """

    print("Start applying diffusion!")

    start_time = time.time()

    newW = [0] * len(aff)

    # First, normalize different networks to avoid scale problems, it is compatible with pandas dataframe
    for n, mat in enumerate(aff):
        # normalize affinity matrix based on strength of edges
        # mat = mat / np.nansum(mat, axis=1, keepdims=True)
        aff[n] = _stable_normalized_pd(mat)
        # aff[n] = check_symmetric(mat, raise_warning=False)

        # apply KNN threshold to normalized affinity matrix
        # We need to crop the intersecting samples from newW matrices
        neighbor_size = min(int(neighbor_size), mat.shape[0])
        newW[n] = _find_dominate_set(aff[n], neighbor_size)

    # If there is only one view, return it
    if len(aff) == 1:
        print("Only one view, return it directly")
        return newW

    for iteration in range(fusing_iteration):

        print(f"\n-- Iteration {iteration + 1} --")

        iteration_time = time.time()

        # Make a copy of the aff matrix for this iteration
        # goal is to update aff[n], but it is the average of all the defused matrices
        # Make a copy of add[n], and set it to 0
        aff_next = []
        for k in range(len(aff)):
            aff_temp = aff[k].copy()
            for col in aff_temp.columns:
                aff_temp[col].values[:] = 0
            aff_next.append(aff_temp)

        for n, mat in enumerate(aff):
            # temporarily convert nans to 0 to avoid propagation errors
            nzW = newW[n]  # TODO: not sure this is a deep copy or not

            for j, mat_tofuse in enumerate(aff):
                if n == j:
                    continue

                # reorder mat_tofuse to have the common samples
                # Optimized (cache values before reuse)
                # Precompute sample order as a NumPy array
                sample_order = np.array(sorted(dicts_common[(j, n)]) + sorted(dicts_unique[(j, n)]))

                # Use .loc for fast reordering instead of reindex
                mat_tofuse = mat_tofuse.loc[sample_order, sample_order]

                # Next, let's crop mat_tofuse
                num_common = len(dicts_common[(n, j)])
                mat_cols = mat_tofuse.columns
                mat_shape = mat_tofuse.shape
                to_drop_mat = mat_cols[num_common: mat_shape[1]].values.tolist()

                mat_tofuse_crop = mat_tofuse.drop(to_drop_mat, axis=1)
                mat_tofuse_crop = mat_tofuse_crop.drop(to_drop_mat, axis=0)

                # Next, add the similarity from the view to fused to the current view identity matrix
                nzW_shape = nzW.shape[0]
                nzW_identity = pd.DataFrame(
                    data=np.identity(nzW_shape),
                    index=original_order[n],
                    columns=original_order[n],
                )

                mat_tofuse_union = nzW_identity + mat_tofuse_crop
                mat_tofuse_union.fillna(0.0, inplace=True)
                mat_tofuse_union = _scaling_normalized_pd(mat_tofuse_union,
                                                          ratio=mat_tofuse_crop.shape[0] / nzW_identity.shape[0])
                mat_tofuse_union = check_symmetric(mat_tofuse_union, raise_warning=False)
                mat_tofuse_union = mat_tofuse_union.reindex(original_order[n], axis=1)
                mat_tofuse_union = mat_tofuse_union.reindex(original_order[n], axis=0)

                # Now we are ready to do the diffusion

                if newW[n].shape[0] < 2000:
                    # print('Using dense matrix...')
                    nzW_T = np.transpose(nzW)

                    start = time.time()
                    aff0_temp = nzW.dot(
                        mat_tofuse_union.dot(nzW_T)
                    )
                    elapsed = time.time() - start
                    print(f"Dense multiplication took {elapsed:.4f} seconds.")

                else:
                    # print('Using sparse matrix...')
                    nzW_sparse = csr_matrix(newW[n].values)
                    nzW_T_sparse = nzW_sparse.transpose()
                    mat_union_dense = mat_tofuse_union.values

                    start = time.time()
                    intermediate = nzW_sparse @ mat_union_dense  # sparse × dense → dense
                    aff0_temp_np = intermediate @ nzW_T_sparse  # dense × sparse → dense
                    elapsed = time.time() - start
                    print(f"Sparse multiplication took {elapsed:.4f} seconds.")

                    aff0_temp = pd.DataFrame(
                        aff0_temp_np,
                        index=original_order[n],
                        columns=original_order[n]
                    )

                #################################################
                # Experimentally introduce a weighting mechanism, use the exponential weight;
                # Already proved it's not a good idea
                # num_com = mat_tofuse_crop.shape[0] / aff[n].shape[0]
                # alpha = pow(2, num_com) - 1
                # aff0_temp = alpha * aff0_temp + (1-alpha) * aff[n]

                # aff0_temp = _B0_normalized(aff0_temp, alpha=normalization_factor)
                aff0_temp = _stable_normalized_pd(aff0_temp)
                # aff0_temp = check_symmetric(aff0_temp, raise_warning=False)

                aff_next[n] = np.add(aff0_temp, aff_next[n])

            aff_next[n] = np.divide(aff_next[n], len(aff) - 1)
            # aff_next[n] = _stable_normalized_pd(aff_next[n])

        # put the value in aff_next back to aff
        for k in range(len(aff)):
            aff[k] = aff_next[k]

        print(f"\nTime for Iteration {iteration + 1}: {time.time() - iteration_time}s")

    for n, mat in enumerate(aff):
        aff[n] = _stable_normalized_pd(mat)
        # aff[n] = check_symmetric(mat, raise_warning=False)

    end_time = time.time()
    print(f"\nDiffusion ends! Times: {end_time - start_time}s")
    return aff
