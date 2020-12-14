import numpy as np
import matplotlib.pyplot as plt
import math
from KMeans import KMeans

def euclidean_distance(x, y):
    return math.sqrt(np.sum((x - y) ** 2))

def silhouette_samples(X, labels):
    """Evaluate the silhouette for each point and return them as a list.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : array, shape = N
    """
    silhoutte_list = []
    for point_id,point in enumerate(X):
        print(point_id)
        dist = []
        cluster_id = labels[point_id]
        cluster_points = X[labels == cluster_id]
        for cluster_point in cluster_points:
            if not np.array_equal(cluster_point,point):
                dist.append(euclidean_distance(point,cluster_point))
        a = 1 / (len(cluster_points) - 1) * np.sum(np.array(dist))

        other_clusters_points = X[labels != cluster_id]
        dist = []
        for other_clusters_point in other_clusters_points:
            dist.append(euclidean_distance(point,other_clusters_point))
        b = np.min(1 / len(other_clusters_points) * np.sum(np.array(dist)))

        s = (b - a) / (max(a, b))
        silhoutte_list.append(s)
    return silhoutte_list


def silhouette_score(silhouette):
    return np.mean(np.array(silhouette))

"""2D Gaussian"""

X_Syn_Gaussian = np.loadtxt('2D_gauss_clusters.txt',delimiter=',',skiprows=1)

plt.scatter(x=X_Syn_Gaussian[:,0],y=X_Syn_Gaussian[:,1],s=10)
plt.show()  # 15 globular clusters at first sight

num_clusters = 15
my_KMeans = KMeans(n_clusters=num_clusters)
labels_Syn_Gaussian = my_KMeans.fit_predict(X_Syn_Gaussian).astype('int32')

"""Chameleon"""

X_Chameleon = np.loadtxt('chameleon_clusters.txt',delimiter=',',skiprows=1)

plt.scatter(x=X_Chameleon[:,0],y=X_Chameleon[:,1],s=10)
plt.show()  # 6 globular clusters at first sight

num_clusters = 6
my_KMeans = KMeans(n_clusters=num_clusters)

labels_Chameleon = my_KMeans.fit_predict(X_Chameleon).astype('int32')

silh_list = silhouette_samples(X_Syn_Gaussian, labels_Syn_Gaussian)
silh_mean = silhouette_score(silh_list)
print(f'Mean silhouettes - 2D Gaussian: {silh_mean}')

silh_list = silhouette_samples(X_Chameleon, labels_Chameleon)
silh_mean = silhouette_score(silh_list)
print(f'Mean silhouettes - Chameleon: {silh_mean}')