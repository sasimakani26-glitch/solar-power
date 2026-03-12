import joblib
import config

def select_best_model(results_df, models):

    best_row = results_df.loc[results_df["RMSE"].idxmin()]
    best_model_name = best_row["Model"]

    best_model = models[best_model_name]
    joblib.dump(best_model, config.MODEL_PATH)

    return best_model_name
