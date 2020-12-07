import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r'NYC_Airbnb/development.csv')

corr = df.corr()
sns.heatmap(corr,annot=True,vmin=-0.4,vmax=1)
plt.tight_layout()
plt.show()

# removing useless features
df.drop(['name','host_name','last_review','id','longitude','reviews_per_month'],axis=1,inplace=True)

# EDA
df.latitude.hist()  # normally distributed
plt.show()
df.minimum_nights.hist()
plt.show()
df.boxplot(column='minimum_nights')
plt.show()
df.boxplot(column='calculated_host_listings_count')
plt.show()
df.availability_365.hist()
plt.show()

# encoding columns
df = pd.get_dummies(df)

y = df.price
X = df.drop(columns=['price'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

reg = RandomForestRegressor(n_estimators=100,max_features='sqrt')
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print(f'R2 score: {r2_score(y_test,y_pred)}')

"""Evaluation"""

df_test = pd.read_csv(r'NYC_Airbnb/evaluation.csv')
ids = df_test.id
df_test.drop(['id','longitude','reviews_per_month'],axis=1,inplace=True)
df_test.drop(['name','host_name','last_review'],axis=1,inplace=True)
df_test = pd.get_dummies(df_test)
missing_columns = set(X) - set(df_test)  # df_test doesn't contain all the neighbourhoods of X, need to add missing columns
for c in missing_columns:
    df_test[c] = 0
df_test = df_test[X.columns]
y_pred = reg.predict(df_test)
df = pd.DataFrame({"Id": ids, "Predicted": y_pred})
df.to_csv("mypredictions.csv",sep=",",index=False)