from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from MyRandomForest import MyRandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1
dataset = fetch_openml("mnist_784")
X = dataset["data"]
y = dataset["target"]

# 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7)
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test,y_pred)}")

# 3 - 4
print('Starting my random forest ...')
n_estimators = 10
p = X_train.shape[1]  # p = # columns
myclf = MyRandomForestClassifier(n_estimators,int(p ** 0.5))
myclf.fit(X_train,y_train)
y_pred = myclf.predict(X_test)
print(f'My random forest, number of estimators: {n_estimators} , accuracy score: {accuracy_score(y_test,y_pred)}')

# 5
clf = RandomForestClassifier(n_estimators=3, max_features=int(784 ** 0.5))
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f'Random forest, number of estimators: {n_estimators} , accuracy score: {accuracy_score(y_test,y_pred)}')

# 6
features_importance = myclf.get_feature_importances()
print(f'Features importance: {features_importance}')

# 7
sns.heatmap(np.reshape(features_importance, (28,28)), cmap='binary')
plt.show()
sns.heatmap(np.reshape(clf.feature_importances_, (28,28)), cmap='binary')
plt.show()