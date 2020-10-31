import numpy as np
import math

class KNearestNeighbors:
    def __init__(self, k, distance_metric="euclidean",weights="uniform"):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.X_test = None


    def fit(self, X, y):
        """
        Store the 'prior knowledge' of you model that will be used
        to predict new labels.
        :param X : input data points, ndarray, shape = (R,C).
        :param y : input labels, ndarray, shape = (R,).
        """
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        """Run the KNN classification on X.
        :param X: input data points, ndarray, shape = (N,C).
        :return: labels : ndarray, shape = (N,).
        """
        self.X_test = X
        predictions = []
        labels = set(self.y_train)  # labels contains the species of flowers

        # I create a 2d-np-array containing distances from a flower of X_test (row) of all X_train flowers(columns)
        euclidean_distances = np.zeros((len(self.X_test),len(self.X_train)))

        for flower_index,flower in enumerate(self.X_test):
            for comparison_flower_index, comparison_flower in enumerate(self.X_train):
                current_distance = self.DISTANCE[self.distance_metric](flower,comparison_flower)  # I switch among functions based on the distance_metric
                euclidean_distances[flower_index,comparison_flower_index] = current_distance

        # 2d np array with shape (len(self.X_test),len(self.X_train)) with indeces of nearest k flowers
        k_neighbors_indeces = euclidean_distances.argsort()[:, :self.k]

        for flower_index, neighbors_indeces in enumerate(k_neighbors_indeces):
            votes = dict((label, 0) for label in labels)  # initialize the votes: {'Iris-Setosa': 0 , 'Iris-Virginica': 0, 'Iris-Versicolor': 0}
            flag_equal_flowers = False
            for index in neighbors_indeces:
                distance = euclidean_distances[flower_index,index]
                if distance == 0:
                    flag_equal_flowers = True
                    predictions.append(self.y_train[index])
                    break
                else:
                    if self.weights != 'distance':
                        votes[self.y_train[index]] += 1 / (euclidean_distances[flower_index,index] ** 2)
                    else:
                        votes[self.y_train[index]] += 1 / (euclidean_distances[flower_index, index])
            if not flag_equal_flowers:
                predictions.append(max(votes, key=votes.get))
        return np.array(predictions)


    def euclidean_distance(p, q):  # p and q are 2d numpy arrays
        return math.sqrt(np.sum((p-q)**2))

    def cosine_distance(p,q):
        num = np.sum(p * q)
        den = math.sqrt(np.sum(p**2)) * math.sqrt(np.sum(q**2))
        return 1 - abs(num/den)

    def manhattan_distance(p,q):
        return np.sum(abs(p-q))

    DISTANCE = {
        'euclidean': euclidean_distance,
        'cosine': cosine_distance,
        'manhattan': manhattan_distance
    }
