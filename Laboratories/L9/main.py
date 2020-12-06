import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r'NYC_Airbnb/development.csv')

# removing rows with no name, host_name, last_review or reviews_per_month
df = df[df.name.isna() == False]
df = df[df.host_name.isna() == False]
df = df[df.last_review.isna() == False]
df = df[df.reviews_per_month != 0]

corr = df.corr()
sns.heatmap(corr,annot=True,vmin=-0.4,vmax=1)
plt.tight_layout()
plt.show()

# removing useless features
df.drop(['id','longitude','number_of_reviews','reviews_per_month'],axis=1,inplace=True)
print(df.corr().price)

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

# removing useless columns
df.drop(['name','host_name','last_review'],axis=1,inplace=True)

# encoding columns
df = pd.get_dummies(df)

y = df.price
X = df.drop(columns=['price'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# reg = make_pipeline(PolynomialFeatures(2),RandomForestRegressor(n_estimators=100,max_features='sqrt'))
reg = RandomForestRegressor(n_estimators=100,max_features='sqrt')
# reg = make_pipeline(PolynomialFeatures(2),Lasso(alpha=1))
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

"""Evaluation"""

df_test = pd.read_csv(r'NYC_Airbnb/evaluation.csv')
ids = df_test.id
df_test.drop(['id','longitude','number_of_reviews','reviews_per_month'],axis=1,inplace=True)
df_test.drop(['name','host_name','last_review'],axis=1,inplace=True)
df_test = pd.get_dummies(df_test)
missing_columns = set(X) - set(df_test)  # df_test doesn't contain all the neighbourhoods of X, need to add missing columns
for c in missing_columns:
    df_test[c] = 0
df_test = df_test[X.columns]
y_pred = reg.predict(df_test)
df = pd.DataFrame({"Id": ids, "Predicted": y_pred})
df.to_csv("mypredictions.csv",sep=",",index=False)