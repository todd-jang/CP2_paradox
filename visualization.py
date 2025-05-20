import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('levelset.csv')

# Select features and target
features = ['a', 'b', 'c', 'n_xy', 'n_xz', 'n_yz', 'c/a', 'b/a', 'n3', 'V', 'r']
target = 'roundness'

# Drop rows with missing values
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)
print(feature_importance)

# Optional: Plot
feature_importance.plot(kind='bar')
plt.ylabel('Importance')
plt.title('Feature Importance for Roundness Prediction')
plt.tight_layout()
plt.show()
