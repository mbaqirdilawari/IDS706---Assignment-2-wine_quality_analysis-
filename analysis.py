import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", shutil.get_terminal_size(fallback=(120, 24)).columns)

# Step 1: Importing the dataset
wine = pd.read_csv("data/wine_quality_merged.csv")
print(f"Total rows: {len(wine)}")
print(wine["type"].value_counts())

print(" " " ")

# Step 2: Inspecting the data
print(f"Shape (rows, columns): {wine.shape}")
print(wine.head())
wine.info()
print(wine.describe())
print("Missing values:")
print(wine.isnull().sum())
print(f"Duplicate rows: {wine.duplicated().sum()}")

print(" " " ")

# Step 3: Filtering

# Filtering for high quality wines (quality >= 7)
high_quality = wine.query("quality >= 7")
print(f"High quality wines: {len(high_quality)} out of {len(wine)}")
print(high_quality[["type", "alcohol", "quality"]].head())

# Filtering for bad quality wines (quality <= 4)
bad_quality = wine.query("quality <= 4")
print(f"Bad quality wines: {len(bad_quality)} out of {len(wine)}")
print(bad_quality[["type", "alcohol", "quality"]].head())

# Filtering for medium quality wines (4 < quality < 7)
medium_quality = wine.query("quality > 4 and quality < 7")
print(f"Medium quality wines: {len(medium_quality)} out of {len(wine)}")
print(medium_quality[["type", "alcohol", "quality"]].head())

# Filtering for red wines with alcohol content greater than 12%
high_alcohol_red = wine.query("type == 'red' and alcohol > 12")
print(f"High alcohol red wines: {len(high_alcohol_red)} out of {len(wine)}")
print(high_alcohol_red[["type", "alcohol", "quality"]].head())

# Filtering for red wines with alcohol content less than 10%
low_alcohol_red = wine.query("type == 'red' and alcohol < 10")
print(f"Low alcohol red wines: {len(low_alcohol_red)} out of {len(wine)}")
print(low_alcohol_red[["type", "alcohol", "quality"]].head())

print(" " " ")

# Step 4: Grouping
by_type = wine.groupby("type").agg(
    avg_alcohol=("alcohol", "mean"),
    std_alcohol=("alcohol", "std"),
    min_alcohol=("alcohol", "min"),
    max_alcohol=("alcohol", "max"),
    avg_quality=("quality", "mean"),
    no_of_wines=("quality", "count"),
)
print(by_type)

by_quality = wine.groupby("quality").agg(
    avg_alcohol=("alcohol", "mean"),
    std_alcohol=("alcohol", "std"),
    min_alcohol=("alcohol", "min"),
    max_alcohol=("alcohol", "max"),
    no_of_wines=("alcohol", "count"),
)
print(by_quality)

print(" " " ")

# Step 5: Machine learning model
print("Machine Learning Model: Predicting Wine Quality")

features = ["alcohol", "volatile acidity", "sulphates"]
X = wine[features]
y = wine["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R-squared: {r2:.3f}")

for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.3f}")

print(" " " ")

# Step 6: Visualization of Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=wine,
    x="quality",
    y="alcohol",
    hue="type",
    palette={"red": "firebrick", "white": "wheat"},
)
plt.title("Alcohol Content by Wine Quality Score")
plt.xlabel("Quality Score")
plt.ylabel("Alcohol (%)")
plt.tight_layout()
plt.savefig("graphs/quality_vs_alcohol.png", dpi=150)
print("Boxplot is saved as quality_vs_alcohol.png")


# Step 7: Visualization of Scatter Plot with Trend Line
plt.figure(figsize=(10, 6))
sns.regplot(
    data=wine,
    x="alcohol",
    y="density",
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"},
)
plt.title("Alcohol Content vs. Density (All Wines)")
plt.xlabel("Alcohol (%)")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("graphs/alcohol_vs_density.png", dpi=150)
print("Scatter plot with trend line saved as alcohol_vs_density.png")
