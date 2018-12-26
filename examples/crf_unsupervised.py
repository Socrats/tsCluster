import numpy as np
from tscluster.crfs.crfsUnsupervised import cluster_ts
from tests.gen_test_datasets import gen_ts

# Parametrization of the voting pool
k = 12
s = k // 2
m = 3
# size_dataset = 5 * 30
iterations = 100
total_error = 0.
total_ri = 0.


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
