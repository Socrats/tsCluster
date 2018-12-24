import numpy as np
from crfs.crfsUnsupervised import cluster_ts

mus = [[40, 45, 100],
       [85, 60, 100],
       [65, 100, 80],
       [55, 120, 120],
       [120, 50, 120]]
var = 64
std = np.sqrt(var)

# Parametrization of the voting pool
k = 18
s = 9
m = 3
size_dataset = 5 * 30
iterations = 100
total_error = 0.
total_ri = 0.


# def count_errors(classes, learnedClasses):
#     uniqueClasses = frozenset(classes)
#     uniqueLearnedClasses = frozenset(learnedClasses)
#
#     corr = zip(classes,
#                learnedClasses)
# # Correspondence vector: corr[x] = (i, j) means that i was associated with j for sequence x
#
#     # Find correspondance between classes and learnedClasses:
#     # The (i, j) correspondence that appears the most means i=j
#     errs = 0
#     corrList = []
#     for i in uniqueClasses:
#         maxCorr = 0
#         for j in uniqueLearnedClasses:
#             errs += corr.count((i, j))
#             if corr.count((i, j)) > maxCorr:
#                 maxCorr = corr.count((i, j))
#                 corrVal = j
#         corrList.append(corrVal)
#
#     for i, j in enumerate(corrList):
#         errs -= corr.count((i, j))
#
#     return errs, corrList


def gen_ts():
    time_series = np.zeros(shape=(30 * 5, 3), dtype=np.float64)
    for i, cluster in enumerate(mus):
        for j, mu in enumerate(cluster):
            time_series[i * 30:(i * 30) + 30, j] = np.random.normal(mu, std, size=30)

    transformed_ts = time_series[:, 1:] - time_series[:, :-1]
    labels = np.arange(transformed_ts.shape[0])
    np.random.shuffle(labels)
    return time_series, labels


def rand_index(inferred_clusters):
    TP = 0.
    TN = 0.
    # first we count the nb of pairs that have correctly been put in the same cluster
    for i in range(size_dataset):
        # define i cluster
        cluster1 = i // 30
        for j in range(i + 1, size_dataset):
            cluster2 = j // 30
            if cluster1 == cluster2:
                if inferred_clusters[i] == inferred_clusters[j]:
                    # True positive
                    TP += 1
            else:
                if inferred_clusters[i] != inferred_clusters[j]:
                    # True negative
                    TN += 1

    return (TP + TN) / ((size_dataset * (size_dataset - 1)) / 2)


if __name__ == "__main__":
    # Calculate the Rand Index
    for i in range(iterations):
        transformed_ts, labels = gen_ts()
        new_labels = cluster_ts(transformed_ts, labels, k, s, m, max_iterations=500)
        ri = rand_index(new_labels)
        error = 1 - ri
        print('iteration {}: (ri {}, error {}, nb_cluster {})'.format(i, ri, error,
                                                                      np.unique(new_labels).shape[0]))
        total_error += error
        total_ri += ri
    print(total_error / iterations)
    print(total_ri / iterations)
