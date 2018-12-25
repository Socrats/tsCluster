import numpy as np


def gen_simple_dataset():
    mus = [[40, 45, 100],
           [85, 60, 100],
           [65, 100, 80],
           [55, 120, 120],
           [120, 50, 120]]
    var = 64
    std = var ** 0.5
    sizes = [30, 30, 30, 30, 30]

    time_series = np.zeros(shape=(30 * 5, 3), dtype=np.float64)
    for i, cluster in enumerate(mus):
        for j, mu in enumerate(cluster):
            time_series[i * 30:(i * 30) + 30, j] = np.random.normal(mu, std, size=30)

    labels = np.arange(time_series.shape[0])
    np.random.shuffle(labels)
    return time_series[labels], labels, get_clusters(labels, len(mus), sizes)


def gen_artbio_dataset():
    # Create test datasets
    nb_clusters = 5
    alpha = np.arange(0.1, 1.0, 1.0/nb_clusters)
    beta = np.arange(2.0, 1.0, -1.0/nb_clusters)
    epsilon = np.random.normal(0, 1, size=5)
    t = np.arange(20)
    sizes = [60, 60, 60, 80, 60]

    def phi(i, t):
        if i == 4:
            return alpha[i] * t + beta[i] + epsilon[i]
        else:
            return np.sin(alpha[i] * t + beta[i]) + epsilon[i]

    ts = np.array([phi(0, t) for _ in range(sizes[0])], dtype=np.float64)
    for i in range(1, len(sizes)):
        ts = np.concatenate((ts, np.array([phi(i, t) for _ in range(sizes[i])], dtype=np.float64)), axis=0)

    ts = np.array(ts, dtype=np.float64)
    transformed_ts = ts[:, 1:] - ts[:, :-1]
    labels = np.arange(transformed_ts.shape[0])
    np.random.shuffle(labels)

    return transformed_ts[labels], labels, get_clusters(labels, len(alpha), sizes)


def get_clusters(labels, nb_clusters, sizes):
    clusters = []
    index = 0
    for i in range(nb_clusters):
        clusters.append(np.where((labels < (index + sizes[i])) & (labels >= index))[0])
        index += sizes[i]
    return clusters


def gen_ts(dataset_name="simple_dataset"):
    if dataset_name == "simple_dataset":
        return gen_simple_dataset()
    elif dataset_name == "artbio_dataset":
        return gen_artbio_dataset()
    else:
        return False
