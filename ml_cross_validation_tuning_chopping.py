import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import numpy as np

# Load the data
df = pd.read_csv('chopping.csv', encoding='utf-8-sig')

# Select features and target
features = ['a', 'b', 'c', 'n_xy', 'n_xz', 'n_yz', 'c/a', 'b/a', 'n3', 'V', 'r']
target = 'roundness'

# Drop rows with missing values
df = df.dropna(subset=features + [target])
X = df[features]
y = df[target]

# Split data (for final test evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and parameter grids
models = {
    'Ridge': {
        'model': Ridge(),
        'params': {
            'alpha': [0.01, 0.1, 1, 10, 100]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10]
        }
    }
}

best_estimators = {}
results = {}

for name, cfg in models.items():
    print(f"\n{name} Model Hyperparameter Search with Cross-Validation")
    grid = GridSearchCV(cfg['model'], cfg['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    print("Best CV score (neg MSE):", grid.best_score_)
    best_estimators[name] = grid.best_estimator_

    # Evaluate on test set
    y_pred = grid.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test set MSE: {mse:.4f}")
    results[name] = {'best_params': grid.best_params_, 'cv_score': grid.best_score_, 'test_mse': mse}

# Summary
print("\nModel Comparison Summary:")
for name, res in results.items():
    print(f"{name}:")
    print(f"  Best Params: {res['best_params']}")
    print(f"  CV Score (neg MSE): {res['cv_score']:.4f}")
    print(f"  Test MSE: {res['test_mse']:.4f}")
