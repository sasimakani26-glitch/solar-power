from modules.data_loader import load_dataset, validate_data
from modules.preprocessing import split_and_scale
from modules.train_models import train_ml_models
from modules.evaluate_models import evaluate_ml_models
from modules.model_selector import select_best_model
import config

df = load_dataset("solar_dataset_multi_year_3years.csv")
df = validate_data(df)

X_train, X_test, y_train, y_test = split_and_scale(df)

models = train_ml_models(X_train, y_train)

results = evaluate_ml_models(models, X_test, y_test)

best_model = select_best_model(results, models)

results.to_csv(config.PERFORMANCE_PATH, index=False)

print("Training Completed")
print(results)
print(f"Best Model: {best_model}")
