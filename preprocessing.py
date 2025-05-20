import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the CSV
df = pd.read_csv('levelset.csv')

# Clean the column names (remove empty/unnamed columns)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Fill missing values (simple strategy, can be improved)
df = df.fillna(df.mean(numeric_only=True))

# Select features and target (predict V)
X = df.drop(columns=['V'])
y = df['V']

# Optional: drop non-numeric or problematic columns
X = X.select_dtypes(include=['float64', 'int64'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
