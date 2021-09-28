# -*- coding: utf-8 -*-
"""
Created on March, 08 2016
@author: Jesus Cid.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipdb

""" Load a set of predictions, labels and labeling metadata and computes
    different ROCs to estiamte different thresholds.

    This is a version of AnalyzeROCs.py that takes all data from dataset.pkl.
    This is not desirable, because the predict values may have been compute
    using all labels and, thus, there is some training bias in the error
    estimates.
"""


def compute_globalstats(rs0al1, relabels):

    print("=============================================")
    print("*** DATASET STATISTICS:")

    n_samples = len(rs0al1)
    print("--- Total size: {0} samples".format(n_samples))
    n_old = np.count_nonzero([r is None for r in rs0al1])
    print("------ Old labels:  {0} samples".format(n_old))
    n_recent = n_samples - n_old
    print("------ Recent labels:  {0} samples".format(n_recent))
    flags = list(zip(relabels, rs0al1))
    n_rec_rs = np.count_nonzero([x == (1, 0) for x in flags])
    n_rec_al = np.count_nonzero([x == (1, 1) for x in flags])
    n_new_rs = np.count_nonzero([x == (0, 0) for x in flags])
    n_new_al = np.count_nonzero([x == (0, 1) for x in flags])
    n_rs = n_rec_rs + n_new_rs
    n_al = n_rec_al + n_new_al
    n_new = n_new_rs + n_new_al
    n_rec = n_rec_rs + n_rec_al

    print("--------- Total Random Sampling: {0} samples".format(n_rs))
    print("--------- Total Active Learning: {0} samples".format(n_al))
    print("--------- Total Recycled: {0} samples".format(n_rec))
    print("--------- Total New:      {0} samples".format(n_new))
    print("-----------------------------------------------")
    print("------------ Recycled, Random Sampling: {0} samples".format(
        n_rec_rs))
    print("------------ Recycled, Active Learning: {0} samples".format(
        n_rec_al))
    print("------------ New, Random Sampling:      {0} samples".format(
        n_new_rs))
    print("------------ New, Active Learning:      {0} samples".format(
        n_new_al))

    return n_samples, n_rs, n_al


def compute_ROC(y_sorted):

    nFN = np.cumsum([max(ys, 0) for ys in y_sorted])
    nTN = np.cumsum([max(-ys, 0) for ys in y_sorted])

    y_sorted0 = np.append(y_sorted, 0)
    nFP = np.cumsum([max(-ys, 0) for ys in y_sorted0[::-1]])[:0:-1]
    nTP = np.cumsum([max(ys, 0) for ys in y_sorted0[::-1]])[:0:-1]

    TPR = nTP.astype(float)/(nTP + nFN)
    FPR = nFP.astype(float)/(nTN + nFP)

    return nFP, nFN, nTN, nTP, TPR, FPR


def compute_tpfn(p, NFP, NFN):

    """ Given two arrays with the cumulative sum of false positive (NFP) and
        false negatives (NFN), compute the threshold
    """

    DIFF = np.abs(NFP - NFN)
    pos = np.argmin(DIFF)
    th = p[pos]

    return th, pos


def plotROCs(p, y, w, rs0al1, relabels, category=None):

    """ Plot three possible decision thresholds for a classifier with
        predictions p for samples with labels y, depending on the use of
        labeling information.

        ARGS:
           :p      :Preditions in the range [-1, 1]
           :y      :Labels in {-1, 1}
           :w      :Non-negative weights.
           :rs0al1 :Type of labelling indicator.
                     = 0 for samples taken with random sampling
                     = 1 for samples taken with active learning
                     None for samples with unknown labeling method.
    """

    # Clean data.
    # Sort all input arrays according to preds.
    orden = np.argsort(np.array(p))
    # Ignore non numeric predictions
    orden = [k for k in orden if pd.notnull(p[k])]
    p_sorted = [p[i] for i in orden]
    y_sorted = [y[i] for i in orden]
    rs0al1_sorted = [rs0al1[i] for i in orden]
    w_sorted = [w[i] for i in orden]

    # Unweighted complete averaging.
    NFP, NFN, NTN, NTP, TPR, FPR = compute_ROC(y_sorted)
    umbral_tpfn, pos = compute_tpfn(p_sorted, NFP, NFN)

    # Averaging with Random Sampling labels only
    ind_rs = np.nonzero([r == 0 for r in rs0al1_sorted])[0]
    p_sorted2 = [p_sorted[i] for i in ind_rs]
    y_sorted2 = np.array([y_sorted[i] for i in ind_rs])
    NFPrs, NFNrs, NTNrs, NTPrs, TPRrs, FPRrs = compute_ROC(y_sorted2)
    umbral_tpfn_rs, pos_rs = compute_tpfn(p_sorted2, NFPrs, NFNrs)

    # Weighted averaging
    wy_sorted = [w_sorted[i] * y_sorted[i] for i in range(len(w_sorted))]
    NFPw, NFNw, NTNw, NTPw, TPRw, FPRw = compute_ROC(wy_sorted)
    umbral_tpfn_w, pos_w = compute_tpfn(p_sorted, NFPw, NFNw)

    # # NFP vs NFN
    # plt.figure()
    # plt.plot(NFP, NFN, 'r', NFPrs, NFNrs, 'g', NFPw, NFNw, 'b')
    # plt.xlabel('Number of False Negatives')
    # plt.ylabel('Number of False Positives')
    # plt.show(block=False)

    # ROC: TPR vs FPR
    plt.figure()
    plt.plot(FPR, TPR, 'r', label='Unweighted')
    plt.plot(FPRrs, TPRrs, 'g', label='Random Sampling')
    plt.plot(FPRw, TPRw, 'b', label='Weighted')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Estimated ROC curve. Category ' + str(category))
    plt.legend(loc=4)
    plt.plot(FPR[pos], TPR[pos], 'r.', markersize=10)
    plt.plot(FPRrs[pos_rs], TPRrs[pos_rs], 'g.', markersize=10)
    plt.plot(FPRw[pos_w], TPRw[pos_w], 'b.', markersize=10)
    plt.show(block=False)

    # ROC based on sklearn (the results are assentially the same)
    # from sklearn.metrics import roc_curve
    # FPR2, TPR2, tt = roc_curve(y_sorted, p_sorted)
    # FPRrs2, TPRrs2, tt = roc_curve(y_sorted2, p_sorted2)
    # FPRw2, TPRw2, tt = roc_curve(y_sorted, p_sorted, sample_weight=w_sorted)
    # plt.figure()
    # plt.plot(FPR2, TPR2, 'r', label='Unweighted')
    # plt.plot(FPRrs2, TPRrs2, 'g', label='Random Sampling')
    # plt.plot(FPRw2, TPRw2, 'b', label='Weighted')
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.title('Estimated ROC curve (sklearn)')
    # plt.legend(loc=4)
    # plt.show(block=False)

    print("=============================================")
    print("*** DATASET ANALYSIS FOR CATEGORY {0}:".format(
        category))

    n_samples = len(y)
    n_old = np.count_nonzero([r is None for r in rs0al1])
    n_recent = n_samples - n_old
    flags = list(zip(relabels, rs0al1))
    n_rec_rs = np.count_nonzero([x == (1, 0) for x in flags])
    n_new_rs = np.count_nonzero([x == (0, 0) for x in flags])
    n_rs = n_rec_rs + n_new_rs

    print("*** DATASET:")
    print("--- No. of labels: {0}".format(n_samples))
    print("--- No. of valid predictions: {0}".format(len(orden)))

    print("*** THRESHOLDS:")
    print("--- All samples: {0}".format(umbral_tpfn))
    print("--- RS:          {0}".format(umbral_tpfn_rs))
    print("--- W:           {0}".format(umbral_tpfn_w))

    print("*** Estimation of CLASS PROPORTIONS:")
    print("--- Based on LABELS ONLY:")
    r_y_all = float((y == 1).sum())/n_samples
    y_recent = np.array(
        [y[n] for n in range(n_samples) if rs0al1[n] in {0, 1} or
         relabels[n] == 1])
    r_y_recent = float((y_recent == 1).sum())/n_recent
    r_y_rs = float((y_sorted2 == 1).sum())/n_rs
    r_y_w = sum([wy for wy in wy_sorted if wy > 0])/np.array(w_sorted).sum()

    print("------ All labels:      {0}".format(r_y_all))
    print("------ Recent labels: {0}".format(r_y_recent))
    print("------ Random sampling: {0}".format(r_y_rs))
    print("------ Weighted: {0}".format(r_y_w))

    print("--- Based on DECISIONS:")
    print("------ FPR (Random Sampling): {0}".format(FPRrs[pos_rs]))
    print("------ TPR (Random Sampling): {0}".format(TPRrs[pos_rs]))
    print("------ Cut point: {0}".format(pos_rs))
    print("------ FPR (Active Learning): {0}".format(FPRw[pos_w]))
    print("------ FPR (Active Learning): {0}".format(TPRw[pos_w]))
    print("------ Cut point: {0}".format(pos_w))
    print("")

    return umbral_tpfn, umbral_tpfn_rs, umbral_tpfn_w
