import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data (example for levelset.csv)
df = pd.read_csv('levelset.csv')

# Select features and target
features = ['a', 'b', 'c', 'n_xy', 'n_xz', 'n_yz', 'c/a', 'b/a', 'n3', 'V', 'r']
target = 'roundness'

# Drop rows with missing target
df = df.dropna(subset=[target])

# Handle missing values in features (simple fill for illustration)
df = df.fillna(df.mean())

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
