import numpy as np
import matplotlib.pyplot as plt
import math

class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def fit_predict(self, X, plot_clusters=False, plot_step=5):
        """
        Run the K-means clustering on X.
        :param X: input data points, array, shape = (N,C).
        :param plot_clusters: decide whether display clusters and centroids sometimes during the run
        :param plot_step: display clusters and centroids every plot_step iterations
        :return: labels : array, shape = N.
        """
        random_centroids_x = np.random.randint(low=np.min(X[:, 0]), high=np.max(X[:, 0]), size=self.n_clusters)
        random_centroids_y = np.random.randint(low=np.min(X[:, 1]), high=np.max(X[:, 1]), size=self.n_clusters)
        random_centroids = np.stack((random_centroids_x,random_centroids_y),axis=1)
        self.labels = np.zeros((len(X),))
        self.centroids = random_centroids
        previous_centroids = np.empty((self.n_clusters,2))
        iteration = 1
        while not np.array_equal(self.centroids,previous_centroids) and iteration <= self.max_iter:
            for point_id in range(len(X)):
                curr_dist = []
                for centroid_id in range(len(self.centroids)):
                    curr_dist.append(self.euclidean_distance(X[point_id],self.centroids[centroid_id]))
                self.labels[point_id] = np.argmin(np.array(curr_dist))

            """Compute new centroids"""
            previous_centroids = self.centroids.copy()
            for centroid_id in range(len(self.centroids)):
                mask_centroid_id = self.labels == centroid_id
                points = X[mask_centroid_id]
                self.centroids[centroid_id] = np.mean(points,axis=0)
            iteration += 1

            if plot_clusters and iteration % plot_step == 0:
                self.print_clusters(X,iteration)

        self.print_clusters(X,iteration)
        return self.labels

    def euclidean_distance(self, x, y):
        return math.sqrt(np.sum((x-y)**2))

    def print_clusters(self, X, iteration):
        plt.scatter(X[:, 0], X[:, 1], s=10, c=self.labels)
        plt.scatter(self.centroids[:,0],self.centroids[:,1],marker='*',c='red')
        plt.title(f'Clustering after {iteration} iterations')
        plt.show()