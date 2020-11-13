from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

class MyRandomForestClassifier:
    def __init__(self, n_estimators, max_features):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.votes = None
        self.classifiers = []

    # train the trees of this random forest using subsets of X (and y)
    def fit(self, X, y):
        indeces_X = np.arange(0, len(X))
        for i in range(self.n_estimators):
            print(f'Training tree number {i}')
            random_row_indeces = np.random.choice(indeces_X, len(X), replace=True)
            Di_X = X[random_row_indeces]  # subset of the dataset, with repetitions
            Di_y = y[random_row_indeces]
            clf = DecisionTreeClassifier(max_features=self.max_features)
            clf.fit(Di_X, Di_y)
            self.classifiers.append(clf)

    # predict the label for each point in X
    def predict(self, X):
        self.votes = pd.DataFrame(np.full((len(X), self.n_estimators), np.nan))
        for i in range(self.n_estimators):
            y_vote = self.classifiers[i].predict(X)
            self.votes[i] = y_vote
        return self.votes.mode(axis=1)[0]

    def get_feature_importances(self):
        tot_features_importance = 0
        for dec_tree_classifier in self.classifiers:
            tot_features_importance += sum(dec_tree_classifier.feature_importances_)
        features_importance = []
        for feature in range(self.max_features ** 2):
            curr_feature_importance = 0
            for dec_tree_classifier in self.classifiers:
                curr_feature_importance += dec_tree_classifier.feature_importances_[feature]
            features_importance.append(curr_feature_importance)
        return features_importance/tot_features_importance