from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
import numpy as np

dataset = load_wine()
X = dataset["data"]
y = dataset["target"]
feature_names = dataset["feature_names"]

# 1
print(f'Records available: {len(X)}')
print(f'Number of missing values in X: {len(X[np.equal(X,None)])} and in y: {len(y[np.equal(y,None)])}')
print(f'Number of elements in X: {X.size} and in y: {y.size}')

# 2
clf = DecisionTreeClassifier()
clf.fit(X,y)

# 3
dot = export_graphviz(clf)
print(dot)

# 4
y_pred = clf.predict(X)
print(accuracy_score(y_pred,y))

# 5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 6
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f'Accuracy score: {round(accuracy_score(y_pred,y_test),2)}')  # around 85-90% , as expected
print(f'Precision score: {round(precision_score(y_pred,y_test,average="micro"),2)}')
print(f'Recall score: {round(recall_score(y_pred,y_test,average="macro"),2)}')
print(f'F1 score: {round(f1_score(y_pred,y_test,average="weighted"),2)}')
print(classification_report(y_pred,y_test,target_names=['Class 0','Class 1','Class 2']))

# 7
params = {"max_depth": [None, 2, 4, 8], "splitter": ["best", "random"]}
best_config = None
best_accuracy = 0
for config in ParameterGrid(params):
    clf = DecisionTreeClassifier(**config)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    curr_accuracy = accuracy_score(y_pred,y_test)
    if curr_accuracy > best_accuracy:
        best_accuracy = curr_accuracy
        best_config = config
print(f'Best configuration: {best_config} with accuracy: {round(best_accuracy,2)}')

# 8
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X,y,test_size=0.2)
kf = KFold(5)  # 5-fold cross-validation
best_accuracy = 0
best_X_train = None
best_y_train = None

for train_indices, validation_indices in kf.split(X_train_valid):
    X_train = X_train_valid[train_indices]
    X_valid = X_train_valid[validation_indices]
    y_train = y_train_valid[train_indices]
    y_valid = y_train_valid[validation_indices]

    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    y_pred_valid = clf.predict(X_valid)
    curr_accuracy = accuracy_score(y_valid, y_pred_valid)
    if curr_accuracy > best_accuracy:
        best_accuracy = curr_accuracy
        best_X_train = X_train
        best_y_train = y_train

clf.fit(best_X_train,best_y_train)
y_pred = clf.predict(X_test)
print(f'Accuracy after k-fold cross-validation: {round(accuracy_score(y_pred,y_test),2)}')