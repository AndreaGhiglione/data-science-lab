import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor


def test_regression(X_train, X_test, y_train, y_test, regressor, reg_name, performances, best_r2, y_best_config_r2):
    reg = regressor
    reg.fit(X_train, y_train)
    y_test_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_test_pred)
    performances.loc[reg_name,"r2"] = r2
    performances.loc[reg_name, "MAE"] = mean_absolute_error(y_test, y_test_pred)
    performances.loc[reg_name, "MSE"] = mean_squared_error(y_test, y_test_pred)
    if y_best_config_r2 is None:
        y_best_config_r2 = y_test_pred
        best_r2 = r2
    else:
        if r2 > best_r2:
            y_best_config_r2 = y_test_pred
            best_r2 = r2
    return best_r2, y_best_config_r2


df = pd.read_csv('WeatherStationLocations.csv')
sensor_id = 22508
sensor_info = df[df['WBAN'] == sensor_id]
print(f"Sensor {sensor_id} is located at ({sensor_info['Latitude'].values[0]},{sensor_info['Longitude'].values[0]})")

# 1
df = pd.read_csv('SummaryofWeather.csv',low_memory=False)

# 2
grouped_df = df.groupby('STA')  # grouping by sensor id
not_null_values_sensors_df = pd.DataFrame(np.zeros((len(grouped_df),2)),index=grouped_df.size().index.values,columns=['Mean Temp', 'Not null data'])
for key,group in grouped_df:
    null_data = group.isna().sum().sum()
    tot_data = np.size(group)
    mean_temp = group['MeanTemp'].mean()
    not_null_values_sensors_df.loc[key] = [mean_temp, tot_data - null_data]
not_null_values_sensors_df['Not null data'] = not_null_values_sensors_df['Not null data'].astype('int64')
not_null_values_sensors_df = not_null_values_sensors_df.sort_values(by=['Not null data'],ascending=False).head(n=10)
print(not_null_values_sensors_df)

with open("WeatherStationLocations.csv") as f:
    locations_df = pd.read_csv(f)
    print(locations_df[locations_df["WBAN"].isin(not_null_values_sensors_df.index)])

# 3
sensor_id = 22508
df = df[df['STA'] == sensor_id][['Date','MeanTemp']]
df['Date'] = df['Date'].astype('datetime64')

# 4
fig = plt.figure(figsize=[12, 5])
plt.plot(df["Date"], df["MeanTemp"])
plt.show()
measurements = df['MeanTemp'].values

# 5
W = 30
T = len(df)
rolling_windows_df = pd.DataFrame(np.zeros((T-W,W+1)))
for t in range(T - W):
    rolling_windows_df.iloc[t] = measurements[t:t+W+1]

# 6
rolling_windows_df.index = df['Date'].values[:len(rolling_windows_df)]
y = rolling_windows_df.iloc[:,-1]
X = rolling_windows_df.iloc[:,:-1]
X_train = X[X.index.year < 1945]
y_train = y[y.index.year < 1945]
X_test = X[X.index.year == 1945]
y_test = y[y.index.year == 1945]

# 7
regressors = {
    "linear": LinearRegression(fit_intercept=True),
    "polynomial": make_pipeline(PolynomialFeatures(2), LinearRegression()),
    "random_forest": RandomForestRegressor(n_estimators=10),
    "Ridge": Ridge(alpha=0.5),
    "Lasso": Lasso(alpha=0.5)
}
metrics = ["r2","MAE","MSE"]
performances = pd.DataFrame(np.zeros((len(regressors), len(metrics))), index=regressors.keys(), columns=metrics)
y_best_r2_pred = None
best_r2_score = None
for r in regressors.keys():
    best_r2_score,y_best_r2_pred = test_regression(X_train, X_test, y_train, y_test, regressors[r], r, performances, best_r2_score, y_best_r2_pred)
print(performances)

# 8
fig,ax = plt.subplots()
ax.plot(X_test[X_test.index.year == 1945].index,y_test,label='Ground truth')
ax.plot(X_test[X_test.index.year == 1945].index,y_best_r2_pred, label='Predictions')
ax.legend()
plt.show()