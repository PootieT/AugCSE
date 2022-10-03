# -*- coding: utf-8 -*-
# source https://github.com/AlexJonesNLP/XLAnalysis5K/blob/main/src/Data%20Generation/svg.py
import numpy as np
import math
import itertools
from tqdm import tqdm


def computeSVG(embs: list) -> list:
    EPS = 1e-10  # For numerical stability with logarithms
    # Computing just the matrices containing the singular values for our embedding spaces
    print("Computing SVDs . . .")
    SVD = [np.linalg.svd(emb, compute_uv=False) for emb in tqdm(embs)]

    for svd in SVD:
        for i in range(svd.shape[0]):
            if math.isclose(svd[i], 0):
                svd[i] = EPS

    # See equation (6) in Section 2.2 of the paper linked above
    print("Computing SVGs . . .")
    return [sum((np.log(a) - np.log(b)) ** 2) for a, b in tqdm(itertools.combinations(SVD, 2))]
