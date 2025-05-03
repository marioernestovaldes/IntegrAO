"""
    The functions in this file is used to get the common and unique samples between any 2 views.

    Author: Shihao Ma at WangLab, Dec. 14, 2020
"""

import numpy as np
from collections import defaultdict


def data_indexing(matrices):
    """
    Performs data indexing on input expression matrices

    Fast version of data_indexing using dictionaries and vectorized operations

    Parameters
    ----------
    matrices : (M, N) array_like
        Input expression matrices, with gene/feature in columns and sample in row.

    Returns
    -------
    matrices_pure: Expression matrices without the first column and first row
    dict_commonSample: dictionaries that give you the common samples between 2 views
    dict_uniqueSample: dictionaries that give you the unique samples between 2 views
    original_order: the original order of samples for each view
    """

    if len(matrices) < 1:
        print("Input nothing, return nothing")
        return None

    print("Start indexing input expression matrices!")

    original_order = [list(df.index) for df in matrices]
    dict_original_order = {i: original_order[i] for i in range(len(matrices))}
    dict_sampleToIndexs = defaultdict(list)

    # Create fast lookup: sample → index in each view
    sample_to_index = [
        {sample: idx for idx, sample in enumerate(df.index)}
        for df in matrices
    ]

    # Populate dict_sampleToIndexs efficiently
    for i, order in enumerate(original_order):
        for idx, sample in enumerate(order):
            dict_sampleToIndexs[sample].append((i, idx))

    dict_commonSample = {}
    dict_commonSampleIndex = {}
    dict_uniqueSample = {}

    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            set_i = set(original_order[i])
            set_j = set(original_order[j])
            common_samples = list(set_i & set_j)

            print(f"Common sample between view{i} and view{j}: {len(common_samples)}")

            dict_commonSample[(i, j)] = common_samples
            dict_commonSample[(j, i)] = common_samples

            dict_commonSampleIndex[(i, j)] = [sample_to_index[i][s] for s in common_samples]
            dict_commonSampleIndex[(j, i)] = [sample_to_index[j][s] for s in common_samples]

            unique_i = list(set_i - set_j)
            unique_j = list(set_j - set_i)

            dict_uniqueSample[(i, j)] = unique_i
            dict_uniqueSample[(j, i)] = unique_j

    return (
        dict_commonSample,
        dict_commonSampleIndex,
        dict_sampleToIndexs,
        dict_uniqueSample,
        original_order,
        dict_original_order,
    )