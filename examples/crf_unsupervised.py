import numpy as np
from crfs.crfsUnsupervised import cluster_ts

mus = [[40, 45, 100],
       [85, 60, 100],
       [65, 100, 80],
       [55, 120, 120],
       [120, 50, 120]]
var = 20
std = np.sqrt(var)

time_series = np.zeros(shape=(30 * 5, 3), dtype=np.float64)
for i, cluster in enumerate(mus):
    for j, mu in enumerate(cluster):
        time_series[i * 30:(i * 30) + 30, j] = np.random.normal(mu, std, size=30)

transformed_ts = time_series[:, 1:] - time_series[:, :-1]
labels = np.arange(transformed_ts.shape[0])
np.random.shuffle(labels)

# Parametrization of the voting pool
k = 18
s = 9
m = 3
size_dataset = len(transformed_ts)
iterations = 100
total_error = 0.
total_ri = 0.


def count_misclassified(inferred_clusters):
    misclassified = 0.
    # first we count the nb of pairs that have correctly been put in the same cluster
    for i in range(size_dataset):
        # define i cluster
        cluster1 = i // 30
        for j in range(i + 1, size_dataset):
            cluster2 = j // 30
            if cluster1 == cluster2:
                if inferred_clusters[i] != inferred_clusters[j]:
                    # True positive
                    misclassified += 1
            else:
                if inferred_clusters[i] == inferred_clusters[j]:
                    # True negative
                    misclassified += 1
    return misclassified / ((size_dataset * (size_dataset - 1))/2)


def rand_index(inferred_clusters):
    TP = 0.
    TN = 0.
    # first we count the nb of pairs that have correctly been put in the same cluster
    for i in range(size_dataset):
        # define i cluster
        cluster1 = i // 30
        for j in range(i+1, size_dataset):
            cluster2 = j // 30
            if cluster1 == cluster2:
                if inferred_clusters[i] == inferred_clusters[j]:
                    # True positive
                    TP += 1
            else:
                if inferred_clusters[i] != inferred_clusters[j]:
                    # True negative
                    TN += 1

    return (TP + TN) / ((size_dataset * (size_dataset - 1))/2)


if __name__ == "__main__":
    # Calculate the Rand Index
    for i in range(iterations):
        new_labels = cluster_ts(transformed_ts, labels, k, s, m, max_iterations=500)
        ri = rand_index(new_labels)
        error = 1-ri
        print('iteration {}: (ri {}, error {})'.format(i, ri, error))
        total_error += error
        total_ri += ri
    print(total_error / iterations)
    print(total_ri / iterations)
