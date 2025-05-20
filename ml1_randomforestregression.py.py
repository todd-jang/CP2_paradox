import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('levelset.csv')

# Select features and target
features = ['a', 'b', 'c', 'n_xy', 'n_xz', 'n_yz', 'c/a', 'b/a', 'n3', 'V', 'r']
target = 'roundness'

# Drop rows with missing values in either features or target
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')


for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {actual:.3f}, Predicted: {pred:.3f}")
