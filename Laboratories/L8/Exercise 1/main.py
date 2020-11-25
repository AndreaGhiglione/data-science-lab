import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

def f1(x):
    return x * np.sin(x) + 2 * x

def f2(x):
    return 10 * np.sin(x) + x ** 2

def f3(x):
    return np.sign(x) * (x ** 2 + 300) + 20 * np.sin(x)

def inject_noise(y):
    """Add a random noise drawn from a normal distribution."""
    return y + np.random.normal(0, 50, size=y.size)

def test_regression(X,y,f,regressor,reg_name,performances,noise):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, shuffle=True)
    y_test = y_test[X_test.argsort()]
    X_test.sort()
    reg = regressor
    reg.fit(X_train[:, np.newaxis], y_train)  # or X_train.reshape(-1,1)
    y_test_pred = reg.predict(X_test[:, np.newaxis])
    plt.title(f"Function {f} with {reg_name} regressor {'with' if noise else 'without'} noise")
    plt.scatter(X_train, y_train, marker='.')
    plt.plot(X_test, y_test_pred)
    plt.show()
    performances.loc[reg_name,"r2"] = r2_score(y_test, y_test_pred)
    performances.loc[reg_name, "MAE"] = mean_absolute_error(y_test, y_test_pred)
    performances.loc[reg_name, "MSE"] = mean_squared_error(y_test, y_test_pred)


functions = {'f1': f1,'f2': f2,'f3': f3}
regressors = {
    "linear": LinearRegression(fit_intercept=True),
    "polynomial": make_pipeline(PolynomialFeatures(5), LinearRegression()),
    "random_forest": RandomForestRegressor(n_estimators=10),
    "Ridge": Ridge(alpha=0.5),
    "Lasso": Lasso(alpha=0.5)
}
metrics = ["r2","MAE","MSE"]

# 1 - 2
tr = 20
n_samples = 100
X = np.linspace(-tr, tr, n_samples)
for f in functions.keys():
    y = functions[f](X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, shuffle=True)
    y_test = y_test[X_test.argsort()]
    X_test.sort()
    plt.title(f"Function {f}")
    plt.scatter(X_train, y_train,marker=".")
    plt.show()

# 3 - 4 - 5
noise = False
for f in functions.keys():
    y = functions[f](X)
    performances = pd.DataFrame(np.zeros((len(regressors),len(metrics))),index=regressors.keys(),columns=metrics)
    for r in regressors.keys():
        test_regression(X,y,f,regressors[r],r,performances,noise)
    print(f"Function {f} without noise:")
    print(performances)
    print()

# 6
noise = True
for f in functions.keys():
    y = inject_noise(functions[f](X))
    performances = pd.DataFrame(np.zeros((len(regressors),len(metrics))),index=regressors.keys(),columns=metrics)
    for r in regressors.keys():
        test_regression(X,y,f,regressors[r],r,performances,noise)
    print(f"Function {f} with noise:")
    print(performances)
    print()

print("f1 is extremely sensitive to noise, while f2 and f3 kept an high level of r2 with regressors which were good without noise")