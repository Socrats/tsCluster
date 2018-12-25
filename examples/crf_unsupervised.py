import numpy as np
from tscluster.crfs.crfsUnsupervised import cluster_ts
from tests.gen_test_datasets import gen_ts

# mus = [[40, 45, 100],
#        [85, 60, 100],
#        [65, 100, 80],
#        [55, 120, 120],
#        [120, 50, 120]]
# var = 64
# std = var ** 0.5
#
# Parametrization of the voting pool
k = 12
s = k // 2
m = 3
# size_dataset = 5 * 30
iterations = 100
total_error = 0.
total_ri = 0.
#
#
# def transform_dataset(ts):
#     return ts
#
#
# def gen_ts():
#     time_series = np.zeros(shape=(30 * 5, 3), dtype=np.float64)
#     for i, cluster in enumerate(mus):
#         for j, mu in enumerate(cluster):
#             time_series[i * 30:(i * 30) + 30, j] = np.random.normal(mu, std, size=30)
#
#     labels = np.arange(time_series.shape[0])
#     np.random.shuffle(labels)
#     return transform_dataset(time_series[labels]), labels, get_clusters(labels)
#
#
# def get_clusters(labels):
#     clusters = []
#     for i in range(nb_clusters):
#         clusters.append(np.where((labels < ((i*30) + 30)) & (labels >= (i*30)))[0])
#     return clusters


def rand_index(inferred_clusters, real_clusters, size_dataset):
    TP = 0.
    TN = 0.
    # first we count the nb of pairs that have correctly been put in the same cluster
    for i in range(size_dataset):
        # define i cluster
        for idx, cluster in enumerate(real_clusters):
            if i in cluster:
                cluster1 = idx
        for j in range(i + 1, size_dataset):
            for idx, cluster in enumerate(real_clusters):
                if j in cluster:
                    cluster2 = idx
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
        np.random.seed()
        transformed_ts, labels, clusters = gen_ts(dataset_name="artbio_dataset")
        new_labels = cluster_ts(transformed_ts, labels, k, s, m, max_iterations=500)

        ri = rand_index(new_labels, clusters, len(transformed_ts))
        error = 1 - ri
        print('iteration {}: (ri {}, error {}, nb_cluster {})'.format(i, ri, error,
                                                                      np.unique(new_labels).shape[0]))
        total_error += error
        total_ri += ri
    print(total_error / iterations)
    print(total_ri / iterations)
