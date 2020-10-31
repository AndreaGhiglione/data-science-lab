import pandas as pd
import numpy as np
from KNN import KNearestNeighbors


""" IRIS DATASET """
df = pd.read_csv('iris.csv',header=None)  # Parse and store the dataset into a pandas DataFrame

df_sampled = df.sample(frac=0.2)  # I take a sample equal to the 20% of the dataframe
X_test = df_sampled.iloc[:,0:4].values  # I select in the DataFrame df_sampled the first 4 columns of all the rows
y_test = df_sampled.loc[:,4].values  # I can even use loc, which works for both for integer labels and not integer ones

X_train = df.iloc[df.index.difference(df_sampled.index)].iloc[:, 0: 4].values  # df_sampled.index gives me the indexes of df_sampled
y_train = df.iloc[df.index.difference(df_sampled.index)].iloc[:,4].values  # first I select the DataFrame of the inversed indexes, then I select the 5th column

dist_metric = 'cosine'
print(f'Distance metric: {dist_metric}')
Knn = KNearestNeighbors(k=5,distance_metric=dist_metric,weights='distance')
Knn.fit(X_train,y_train)
y_pred = Knn.predict(X_test)
counter = 0
for flower,prediction in zip(y_test,y_pred):
    if flower == prediction:
        counter += 1
print(f'The number of correct predictions in Iris dataset was: {counter}/{len(y_test)} with accuracy of {round(counter*100/len(y_test),2)}%')


""" MNIST DATASET """
df = pd.read_csv('mnist_test.csv',header=None)
df_Y = df.iloc[:,0]
df_X = df.iloc[:,1:]  # I remove the first column (contains the digit)
df_sampled_columns_X = df_X.sample(n=100,axis="columns")  # or axis=1 , sampling by columns (100 pixels)
df_sampled_X = df_sampled_columns_X.sample(n=1000,axis="index")  # or axis=0 , sampling by rows (default, like with Iris dataset exercise)

X_test = df_sampled_X.values  # dataframe.values = numpy array
df_differences_X = df_sampled_columns_X.iloc[df_sampled_columns_X.index.difference(df_sampled_X.index)]
X_train = df_differences_X.values

y_test = np.array([df_Y[i] for i in df_sampled_X.index])
y_train = np.array([df_Y[i] for i in df_differences_X.index])

Knn = KNearestNeighbors(k=5,distance_metric='euclidean',weights='uniform')
Knn.fit(X_train,y_train)
y_pred = Knn.predict(X_test)
for digit,prediction in zip(y_test,y_pred):
    if digit == prediction:
        counter += 1
print(f'The number of correct predictions in Mnist dataset was: {counter}/{len(y_test)} with accuracy of {round(counter*100/len(y_test),2)}%')

print('As we could expect, predictions on Iris were (mostly) more accurate than Mnist, because on Mnist we selected a pretty small sample over the total dataset')