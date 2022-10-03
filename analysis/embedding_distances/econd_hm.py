# -*- coding: utf-8 -*-
# source: https://github.com/AlexJonesNLP/XLAnalysis5K/blob/main/src/Data%20Generation/econd_hm.py
import numpy as np
import math
import itertools
from tqdm import tqdm

EPS = 1e-10  # Used in computation


def computeECOND_HM(embs: list) -> list:
    """
    See Section 2.1 of "The Secret is in the Spectra" for a mathematical description of this metric,
    although the implementation below should be easy enough to follow
    """

    # Obtaining the singular values of each of the bitext embeddings,
    # as done in computing the SVG
    print("Computing SVDs . . .")
    SVD = [np.linalg.svd(emb, compute_uv=False) for emb in tqdm(embs)]

    for svd in SVD:
        for i in range(svd.shape[0]):
            if math.isclose(svd[i], 0):
                svd[i] = EPS

    print("Computing ECOND-HMs . . .")
    out = []
    for a, b in tqdm(itertools.combinations(SVD, 2)):
        # Computing the entropy of the normalized singular value distribution of
        # each monolingual embedding space
        norm_a = a / sum(a)
        norm_b = b / sum(b)
        norm_a += EPS
        norm_b += EPS
        ent_a = -sum(norm_a * np.log(norm_a))
        ent_b = -sum(norm_b * np.log(norm_b))
        # Computing the effective rank of matrices a and b (see equation (2) in Section 2.1)
        erank_a = int(np.floor(np.exp(ent_a)))
        erank_b = int(np.floor(np.exp(ent_b)))
        # Computing the effective condition number of a and b (see equatino (4) in Section 2.1)
        econd_a = a[0] / a[erank_a - 1]
        econd_b = b[0] / b[erank_b - 1]
        # Computing the harmonic mean of the effective conditional numbers of a and b
        # (ECOND-HM)
        econd_hm = (2 * econd_a * econd_b) / (econd_a + econd_b)

        out.append(econd_hm)
    return out
