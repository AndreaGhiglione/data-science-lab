import numpy as np
import matplotlib.pyplot as plt
from KMeans import KMeans

"""2D Gaussian"""

X_Syn_Gaussian = np.loadtxt('2D_gauss_clusters.txt',delimiter=',',skiprows=1)

plt.scatter(x=X_Syn_Gaussian[:,0],y=X_Syn_Gaussian[:,1],s=10)
plt.show()  # 15 globular clusters at first sight

num_clusters = 15
my_KMeans = KMeans(n_clusters=num_clusters)
labels_Syn_Gaussian = my_KMeans.fit_predict(X_Syn_Gaussian)

"""Chameleon"""

X_Chameleon = np.loadtxt('chameleon_clusters.txt',delimiter=',',skiprows=1)

plt.scatter(x=X_Chameleon[:,0],y=X_Chameleon[:,1],s=10)
plt.show()  # 6 globular clusters at first sight

num_clusters = 6
my_KMeans = KMeans(n_clusters=num_clusters)

labels_Chameleon = my_KMeans.fit_predict(X_Chameleon)