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
df.boxplot(rot = 45)
plt.title('Boxplot of California Housing Dataset') 
plt.ylabel('Values')
plt.xlabel('Features')
plt.tight_layout()
plt.savefig("figs/boxplot.png", dpi=300, bbox_inches='tight')
plt.show()