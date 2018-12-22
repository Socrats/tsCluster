import numpy as np
from crfs.crfsUnsupervised import cluster_ts

mus = [[40, 45, 100],
        [85, 60, 100],
        [65, 100, 80],
        [55, 120, 120],
        [120, 50, 120]]
var = 64
std = np.sqrt(var)

time_series = np.zeros(shape=(30*5,3), dtype=np.float64)
for i, cluster in enumerate(mus):
    for j, mu in enumerate(cluster):
        time_series[i*30:(i*30)+30, j] = np.random.normal(mu, std, size=30)

transformed_ts = time_series[:, 1:] - time_series[:, :-1]
labels = np.arange(transformed_ts.shape[0])
np.random.shuffle(labels)

# Parametrization of the voting pool
k = 18
s = 1
m = 1

clusters = cluster_ts(transformed_ts, labels, k, s, m)

print(np.unique(clusters).shape[0])
print(clusters)
