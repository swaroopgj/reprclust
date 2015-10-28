# -*- coding: utf-8 -*-
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the module for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Module to compute stability estimates of clustering solutions using a method
inspired by Lange et al. (2004), and Yeo et al. (2011). The procedure
cross-validates across subjects in the following way: the datasets are divided
into training and test set; clustering is performed on the training set, and
the solution is predicted on the test set. Then, the clustering solution is
computed on the test set, and this is compared to the predicted one using the
Adjusted Rand Index, the Adjusted Mutual Information, and the Instability Score
(Lange et. al, 2004) as metrics. Clustering solutions are swept from k=2 to a
maximum k defined by the user.

At the moment the following clustering algorithms are implemented:
    - k-means
    - Gaussian Mixture Models
    - Ward Clustering (structured and unstructured)
    - complete linkage with correlation distance

References:
-----------
Lange, T., Roth, V., Braun, M. and Buhmann J. (2004)
"Stability-based validation of clustering solutions."
Neural computation 16, no. 6 (2004): 1299-1323.

Thomas Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D.,
Hollinshead, M., et al. (2011).
"The organization of the human cerebral cortex estimated by intrinsic
functional connectivity."
Journal of Neurophysiology, 106(3), 1125–1165. doi:10.1152/jn.00338.2011
"""
import copy

from joblib import Parallel, delayed

import numpy as np

from reprclust.cluster_metrics import ARI, AMI

# this must be outside to allow parallelization
def _run_fold(data, train, test, cluster_method, ks, stack=False,
              ground_truth=None, cluster_metrics=(ARI(), AMI())):
    """Run reproducibility algorithm on one fold for all the ks"""
    # initialize methods
    cm_train = cluster_method
    cm_test = copy.deepcopy(cm_train)

    # XXX: this should change depending on the type of data
    data_train = [data[tr_idx] for tr_idx in train]
    data_test = [data[te_idx] for te_idx in test]

    if stack:
        data_train = np.vstack(data_train)
        data_test = np.vstack(data_train)
    else:
        data_train = np.mean(np.dstack(data_train), axis=-1)
        data_test = np.mean(np.dstack(data_test), axis=-1)

    # allocate storing dictionary
    result_fold = {}
    for metric in cluster_metrics:
        result_fold[str(metric)] = np.vstack((ks, np.zeros(len(ks))))
        if ground_truth is not None:
            result_fold[str(metric) + '_gt'] = \
                np.vstack((ks, np.zeros(len(ks))))

    for i_k, k in enumerate(ks):
        # Step 1. Clustering on training/test set and prediction
        # cluster on training set
        cm_train.train(data_train, k, compute_full=True)
        # cluster on test set
        cm_test.train(data_test, k, compute_full=True)

        # predict
        predicted_label = cm_train.predict(data_test, k)
        test_label = cm_test.predict(data_test, k)

        # Step 2. Compute scores and store them
        for metric in cluster_metrics:
            result_fold[str(metric)][1, i_k] = \
                metric(predicted_label, test_label, data=data_test, k=k)
            if ground_truth is not None:
                result_fold[str(metric) + '_gt'][1, i_k] = \
                    metric(predicted_label, ground_truth, data=data_test, k=k)
    return result_fold


def reproducibility(data, splitter, cluster_method, ks, ground_truth=None,
                    stack=False, cluster_metrics=(ARI(), AMI()),
                    n_jobs=1, verbose=51):
    """
    Parameters
    ----------
    data: Input dataset of which rows are clustered.
    splitter: A generator that provides training and testing splits
                over which to cross-validate.
    cluster_method: Cluster method to be used. Refer to `cluster_methods`
                for available options.
    ks: A list of cluster sizes to be computed.
    """

    if not isinstance(ks, (list, np.ndarray)):
        raise ValueError('ks must be a list or numpy array')
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
    fold = delayed(_run_fold)
    results = parallel(fold(data, train, test, cluster_method, ks,
                            ground_truth=ground_truth,
                            stack=stack,
                            cluster_metrics=cluster_metrics)
                       for train, test in splitter)

    scores = {}
    # store everything together now
    for metric in results[0]:
        scores[metric] = np.hstack((res[metric] for res in results))

    return scores
