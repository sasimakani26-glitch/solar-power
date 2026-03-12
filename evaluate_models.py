from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

def evaluate_ml_models(models, X_test, y_test):

    results = []

    for name, model in models.items():

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append([name, mae, rmse, r2])

    return pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
