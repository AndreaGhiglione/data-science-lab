import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_graphviz

# 1
df = pd.read_csv('2d-synthetic.csv')
plt.scatter(x=df['x0'],y=df['x1'],c=np.where(df['label'] == 0,'b','r'))
plt.show()

# 2
clf = DecisionTreeClassifier()
clf.fit(df[['x0','x1']],df['label'])
dot = export_graphviz(clf)
print(dot)