import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file (skip the BOM and treat double-commas as missing values)
df = pd.read_csv('chopping.csv', encoding='utf-8-sig')

# Display the first few rows to understand the data
print(df.head())

# Example 1: Basic statistics for all numeric columns
print(df.describe())

# Example 2: Scatter plot of roundness vs. major axis (a)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='a', y='roundness')
plt.title('Roundness vs. Major Axis (a)')
plt.xlabel('a [mm]')
plt.ylabel('Roundness')
plt.grid(True)
plt.tight_layout()
plt.savefig('roundness_vs_a.png')
plt.show()

# Example 3: Histogram of Volume (V)
plt.figure(figsize=(8, 5))
sns.histplot(df['V'], bins=20, kde=True)
plt.title('Distribution of Volume (V)')
plt.xlabel('Volume [mm³]')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('volume_histogram.png')
plt.show()

# Example 4: Pairplot for selected columns
sns.pairplot(df[['a', 'b', 'c', 'V', 'roundness']].dropna())
plt.suptitle('Pairwise Relationships', y=1.02)
plt.tight_layout()
plt.savefig('pairplot.png')
plt.show()
