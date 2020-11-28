import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

def test_regression(X,y,regressor,reg_name,performances):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, shuffle=True)
    reg = regressor
    reg.fit(X_train, y_train)  # or X_train.reshape(-1,1)
    y_test_pred = reg.predict(X_test)
    performances.loc[reg_name,"r2"] = r2_score(y_test, y_test_pred)
    performances.loc[reg_name, "MAE"] = mean_absolute_error(y_test, y_test_pred)
    performances.loc[reg_name, "MSE"] = mean_squared_error(y_test, y_test_pred)

regressors = {
    "linear": LinearRegression(fit_intercept=True),
    "polynomial": make_pipeline(PolynomialFeatures(2), LinearRegression()),
    "random_forest": RandomForestRegressor(n_estimators=10),
    "Ridge": Ridge(alpha=0.5),
    "Lasso": Lasso(alpha=0.5)
}
metrics = ["r2","MAE","MSE"]

performances = pd.DataFrame(np.zeros((len(regressors),len(metrics))),index=regressors.keys(),columns=metrics)
X, y = make_regression(n_samples=2000, random_state=42, n_features=120,n_informative=5, noise=5)
for r in regressors.keys():
    test_regression(X, y, regressors[r], r, performances)
print(performances)