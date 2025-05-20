import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from tensorflow import keras
from tensorflow.keras import layers

# Load data
df = pd.read_csv('chopping.csv', encoding='utf-8-sig')

features = ['a', 'b', 'c', 'n_xy', 'n_xz', 'n_yz', 'c/a', 'b/a', 'n3', 'V', 'r']
target = 'roundness'
df = df.dropna(subset=features + [target])

# Binarize target using median
threshold = df[target].median()
df['roundness_bin'] = (df[target] > threshold).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df['roundness_bin'], train_size=0.7, random_state=42, stratify=df['roundness_bin']
)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
rf_auc = auc(rf_fpr, rf_tpr)

# DEX-style Deep Neural Network
model = keras.Sequential([
    layers.Input(shape=(len(features),)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Train DEX
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
dex_probs = model.predict(X_test).flatten()
dex_fpr, dex_tpr, _ = roc_curve(y_test, dex_probs)
dex_auc = auc(dex_fpr, dex_tpr)

# Plot ROC
plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f"RandomForest (AUC={rf_auc:.2f})")
plt.plot(dex_fpr, dex_tpr, label=f"DEX (Deep NN, AUC={dex_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: RandomForest vs Deep Expectation Model')
plt.legend()
plt.tight_layout()
plt.savefig('roc_auc_rf_vs_dex.png')
plt.show()
