# distutils: language=c++
"""
Implements the Unsupervised conditional random fields algorithm at
Li, C.T., Yuan, Y. and Wilson, R., 2008. An unsupervised conditional
random fields approach for clustering gene expression time series.
Bioinformatics, 24(21), pp.2467-2473.

crfsUnsupervised. CRFs unsupervised clustering of time-series
    Copyright © 2018  Elias Fernández <eliferna@vub.be>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import numpy as np
cimport numpy as np
from libcpp cimport bool

ctypedef np.float_t DTYPE_t
ctypedef np.int_t INTYPE_t


# cdef np.ndarray[DTYPE_t] euclidean_dist(np.ndarray[DTYPE_t] t1, np.ndarray[DTYPE_t, ndim=2] t2):
#     return np.sqrt(np.sum((t1-t2)**2, axis=1))


cdef np.ndarray[INTYPE_t] update_voting_pool(np.ndarray[DTYPE_t, ndim=2] pool, np.ndarray[INTYPE_t] voting_pool,
                                             np.ndarray[INTYPE_t] ts_index, int focal, int k, int s):
    """
    The voting pool is formed by selecting s MS time series, 1 MD
    and k-s-1 time series selcted at random
    """
    assert k > 1
    assert s > 0
    cdef np.ndarray[DTYPE_t] distances

    # First we get k-s-1 random time-series
    voting_pool[s:-1] = np.random.choice(
        ts_index[(ts_index != voting_pool[-1]) & np.isin(ts_index, voting_pool[:s], invert=True)],
        size=k - s - 1, replace=False)

    # Then we find s most similar (MS) time-series and the most differnt (MD)
    distances = np.linalg.norm(pool[focal] - pool[voting_pool], axis=1)
    voting_pool[:] = voting_pool[distances.argsort()]

    return voting_pool

cdef INTYPE_t check_labels(np.ndarray[DTYPE_t, ndim=2] pool, np.ndarray[INTYPE_t] labels,
                           np.ndarray[INTYPE_t] voting_pool, int m, DTYPE_t D):
    """
    Calculate the potentials of each member of the voting pool with respect to the focal player
    """
    cdef int i, j
    cdef int nvpool = len(voting_pool)
    cdef np.ndarray[INTYPE_t] unique_labels = np.unique(labels)
    cdef int nulabels = len(unique_labels)
    cdef np.ndarray[DTYPE_t] costs = np.zeros(shape=(nulabels,))
    cdef np.ndarray[DTYPE_t] distances

    # Calculate the costs of each label
    for index1 in range(nulabels):
        for i in voting_pool[labels == unique_labels[index1]]:
            distances = np.linalg.norm(pool[i] - pool[voting_pool], axis=1) - D
            for index2, j in enumerate(voting_pool):
                if i == j: continue
                costs[index1] += distances[index2] if (unique_labels[index1] == labels[index2]) else -distances[index2]
    return unique_labels[np.argmin(costs)]

cpdef np.ndarray[INTYPE_t] cluster_ts(np.ndarray[DTYPE_t, ndim=2] ts, np.ndarray[INTYPE_t] labels, int k, int s, int m,
                                      int max_iterations=50, bool inplace=False):
    """
    ts: time-series samples
    k: size of the voting pool
    s: number of most similar players selected for the voting pool
    m: number of time-series to which the distance is compared
    """
    cdef int i = 0;
    cdef int iterations = 0;
    cdef int new_label;
    cdef int nts = len(ts)
    cdef DTYPE_t d = 0.
    cdef DTYPE_t D = 0.
    cdef bool label_updated = True;
    cdef np.ndarray[INTYPE_t, ndim=2] voting_pool = np.zeros(shape=(nts, k), dtype=np.int64)
    cdef np.ndarray[INTYPE_t] ts_index = np.arange(nts)

    if inplace:
        labels_copy = labels
    else:
        labels_copy = labels.copy()

    for i in range(nts):
        voting_pool[i, :] = np.random.choice(ts_index[ts_index != i], size=k, replace=False)
        # get m random members
        tmp = np.random.choice(voting_pool[voting_pool != i], size=m, replace=False)
        dst = np.linalg.norm(ts[i] - ts[tmp], axis=1)
        d += np.min(dst) + np.max(dst)
    D = d / (2. * nts)

    while label_updated:
        # print('[', iterations, '] label 0: ', labels_copy[0])
        label_updated = False
        # First let's calculate the potentials
        for i in range(nts):
            # create the voting pool
            voting_pool[i, :] = update_voting_pool(ts, voting_pool[i], ts_index, i, k, s)
            # Calculate the cost functions of the labels of the members in the voting pool
            new_label = check_labels(ts, labels_copy[voting_pool[i]], voting_pool[i], m, D)
            # Check if the label has been updated
            if new_label != labels_copy[i]:
                label_updated = True
                labels_copy[i] = new_label
        iterations += 1
        if iterations % 10 == 0:
            print('Iteration: ', iterations)
        if iterations >= max_iterations: break

    return labels_copy
