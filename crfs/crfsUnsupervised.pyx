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

# distutils: language=c++
import numpy as np
cimport numpy as np
from libcpp cimport bool
DTYPE = np.float64
ctypedef np.float_t DTYPE_t
ctypedef np.int_t INTYPE_t


cdef np.ndarray[INTYPE_t] get_voting_pool(np.ndarray[DTYPE_t, ndim=2] pool, np.ndarray[INTYPE_t] MS, INTYPE_t MD, int focal, int k, int s):
    """
    The voting pool is formed by selecting s MS time series, 1 MD
    and k-s-1 time series selcted at random
    """
    assert k > 1
    assert s > 0

    ts_index = np.arange(pool.shape[0])

    # First we get k-s-1 random time-series
    voting_pool = np.random.choice(ts_index[(ts_index != MD) & (ts_index != focal) & np.isin(ts_index, MS, invert=True)], size=k-s-1, replace=False)
    voting_pool = np.append([MD], np.append(MS, voting_pool))

    # Then we find s most similar (MS) time-series and the most differnt (MD)
    distances = np.linalg.norm(pool[focal] - pool[voting_pool], axis=1)
    MS[:] = voting_pool[distances.argsort()[:s]]
    MD = voting_pool[np.argmax(distances)]
    voting_pool = np.append(voting_pool, [focal])

    return voting_pool

cdef INTYPE_t check_labels(np.ndarray[DTYPE_t, ndim=2] pool, np.ndarray[INTYPE_t] labels, np.ndarray[INTYPE_t] voting_pool, int m):
    """
    Calculate the potentials of each member of the voting pool with respect to the focal player
    """
    cdef int i,j
    cdef int nvpool = voting_pool.shape[0]
    cdef DTYPE_t dp = 0.
    cdef DTYPE_t do = 0.
    cdef DTYPE_t D = 0.
    cdef np.ndarray[DTYPE_t] costs = np.zeros(shape=(nvpool,));
    cdef np.ndarray[INTYPE_t] indexes;

    # First we infer D
    for i in voting_pool:
        # get m random members
        tmp = np.random.choice(voting_pool, size=m, replace=False)
        dst = np.linalg.norm(pool[i] - pool[tmp], axis=1)
        dp += np.min(dst)
        do += np.max(dst)
    D = (dp + do) / DTYPE(2. * nvpool)

    # Calculate the costs
    for index1, i in enumerate(voting_pool):
        distances = np.linalg.norm(pool[i] - pool[voting_pool], axis=1) - D
        for index2, j in enumerate(voting_pool):
            if i==j: continue
            costs[index1] += distances[index2] if (labels[i] == labels[j]) else -distances[index2]
    return labels[voting_pool[np.argmin(costs)]]

cpdef np.ndarray[INTYPE_t] cluster_ts(np.ndarray[DTYPE_t, ndim=2] ts, np.ndarray[INTYPE_t] labels, int k, int s, int m, bool inplace=False):
    """
    ts: time-series samples
    k: size of the voting pool
    s: number of most similar players selected for the voting pool
    m: number of time-series to which the distance is compared
    """
    cdef int i=0;
    cdef int iterations=0;
    cdef int new_label;
    cdef int nts = ts.shape[0]
    cdef bool label_updated = True;
    cdef DTYPE_t dp = 0.
    cdef DTYPE_t do = 0.
    cdef DTYPE_t D = 0.
    cdef np.ndarray[INTYPE_t, ndim=1] voting_pool;
    cdef np.ndarray[INTYPE_t, ndim=2] MS = np.zeros(shape=(nts, s), dtype=np.int64);
    cdef np.ndarray[INTYPE_t, ndim=1] MD = np.zeros(shape=(nts,), dtype=np.int64);
    if inplace:
        labels_copy = labels
    else:
        labels_copy = labels.copy()

    # First we initialize MS and MD
    ts_index = np.arange(nts)
    for i in range(nts):
        tmp = np.random.choice(ts_index[ts_index != i], size=s+1, replace=False)
        MS[i, :] = tmp[:s]
        MD[i] = tmp[-1]

    while label_updated:
        label_updated = False
        # First let's calculate the potentials
        for i in range(nts):
            # create the voting pool
            voting_pool = get_voting_pool(ts, MS[i, :], MD[i], i, k, s)
            MD[i] = voting_pool[0]
            MS[i, :] = voting_pool[1:s+1]
            # Calculate the cost functions of the labels of the members in the voting pool
            new_label = check_labels(ts, labels_copy, voting_pool, m)
            # Check if the label has been updated
            if new_label != labels_copy[i]:
                label_updated = True
                labels_copy[i] = new_label
        iterations += 1
        if iterations % 10 == 0:
            print('Iteration: ',iterations)
        if iterations >= 50: break

    return labels_copy