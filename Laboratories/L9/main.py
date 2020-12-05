import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso,RidgeCV,LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats

def encode_strings(strings):
    dict = {}
    id = 0
    for string in strings:
        dict[string] = id
        id = id+1
    return dict

df = pd.read_csv(r'NYC_Airbnb/development.csv')

# removing rows with no name, host_name, last_review or reviews_per_month
df = df[df.name.isna() == False]
df = df[df.host_name.isna() == False]
df = df[df.last_review.isna() == False]
df = df[df.reviews_per_month != 0]

corr = df.corr()
sns.heatmap(corr,annot=True,vmin=-0.4,vmax=1)
plt.show()

# removing useless features
df.drop(['id','name','host_name','last_review'],axis=1,inplace=True)

# EDA

thrsehold_outliers = 1000  # removing outliers for plotting
prices = df[df.price < thrsehold_outliers].price.values
sns.displot(prices,kde=True,height=5,aspect=1.9)
plt.show()

df.boxplot(column='price',by='room_type')
plt.show()
df.boxplot(column='price',by='neighbourhood_group',rot=90)
plt.show()
df.plot.scatter(x='reviews_per_month',y='price')
plt.show()
df.plot.scatter(x='calculated_host_listings_count',y='price')
plt.show()
df.plot.scatter(x='minimum_nights',y='price')
plt.show()
df.plot.scatter(x='availability_365',y='price')
plt.show()
df.plot.scatter(x='room_type',y='price')
plt.show()
df.plot.scatter(x='longitude',y='latitude',c='price',cmap='cool',alpha=0.5)
plt.show()
df[df.price < 300].plot.scatter(x='longitude',y='latitude',c='price',cmap='cool',alpha=0.5)
plt.show()

# removing useless columns
df.drop(['minimum_nights','reviews_per_month','calculated_host_listings_count','availability_365'],axis=1,inplace=True)

sns.heatmap(df.corr(),cmap="seismic",annot=True,vmin=-1,vmax=1)
plt.show()

df = df[df['price'] <= 500]

sns.displot(df['price'], kde=True)
fig = plt.figure()
res = stats.probplot(df['price'], plot=plt)
print("Skewness: %f" % df['price'].skew())
print("Kurtosis: %f" % df['price'].kurt())

# encoding columns
room_types = df.room_type.unique()
neighbourhood_groups = df.neighbourhood_group.unique()
neighbourhoods = df.neighbourhood.unique()
room_types_dict = encode_strings(room_types)
neighbourhood_groups_dict = encode_strings(neighbourhood_groups)
neighbourhoods_dict = encode_strings(neighbourhoods)
df.neighbourhood_group = df.neighbourhood_group.apply(lambda x: neighbourhood_groups_dict[x])
df.room_type = df.room_type.apply(lambda x: room_types_dict[x])
df.neighbourhood = df.neighbourhood.apply(lambda x: neighbourhoods_dict[x])

y = df.price
X = df.drop(columns=['price'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

reg = make_pipeline(PolynomialFeatures(2),RandomForestRegressor())
# reg = RandomForestRegressor()
# reg = make_pipeline(PolynomialFeatures(2),Lasso(alpha=1))
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print(r2_score(y_test, y_pred))


"""Evaluation"""

df = pd.read_csv(r'NYC_Airbnb/evaluation.csv')
ids = df.id
df.drop(['id','name','host_name','last_review'],axis=1,inplace=True)
df.drop(['minimum_nights','reviews_per_month','calculated_host_listings_count','availability_365'],axis=1,inplace=True)

# encoding columns
room_types = df.room_type.unique()
neighbourhood_groups = df.neighbourhood_group.unique()
neighbourhoods = df.neighbourhood.unique()
room_types_dict = encode_strings(room_types)
neighbourhood_groups_dict = encode_strings(neighbourhood_groups)
neighbourhoods_dict = encode_strings(neighbourhoods)
df.neighbourhood_group = df.neighbourhood_group.apply(lambda x: neighbourhood_groups_dict[x])
df.room_type = df.room_type.apply(lambda x: room_types_dict[x])
df.neighbourhood = df.neighbourhood.apply(lambda x: neighbourhoods_dict[x])

y_pred = reg.predict(df)
df = pd.DataFrame({"Id": ids, "Predicted": y_pred})
df.to_csv("mypredictions.csv",sep=",",index=False)