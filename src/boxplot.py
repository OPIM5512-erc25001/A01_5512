from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Create boxplot
plt.figure(figsize=(12, 6))
df.boxplot()
plt.title('California Housing Dataset - Boxplot')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()