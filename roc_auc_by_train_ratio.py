import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# Load data
df = pd.read_csv('chopping.csv', encoding='utf-8-sig')

# Features and target
features = ['a', 'b', 'c', 'n_xy', 'n_xz', 'n_yz', 'c/a', 'b/a', 'n3', 'V', 'r']
target = 'roundness'

# Drop missing
df = df.dropna(subset=features + [target])

# Binarize target using median (you can adjust threshold as needed)
threshold = df[target].median()
df['roundness_bin'] = (df[target] > threshold).astype(int)

train_sizes = [0.5, 0.6, 0.7, 0.8]
plt.figure(figsize=(8, 6))

for train_size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df['roundness_bin'], train_size=train_size, random_state=42, stratify=df['roundness_bin'])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f"Train={int(train_size*100)}% (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve by Training Size')
plt.legend()
plt.tight_layout()
plt.savefig('roc_auc_by_train_ratio.png')
plt.show()
