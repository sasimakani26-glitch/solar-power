from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

def train_ml_models(X_train, y_train):

    models = {}

    models["Linear Regression"] = LinearRegression().fit(X_train, y_train)
    models["Decision Tree"] = DecisionTreeRegressor().fit(X_train, y_train)
    models["Random Forest"] = RandomForestRegressor().fit(X_train, y_train)

    return models
