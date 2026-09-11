import shutil

import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
# Detect the terminal width ourselves: pandas' own auto-detection
# (display.width=None) doesn't always find it, which leaves wide tables
# unwrapped and at the mercy of the terminal's raw line-wrapping (breaking
# column headers and values apart mid-word).
pd.set_option("display.width", shutil.get_terminal_size(fallback=(120, 24)).columns)


def show(df: pl.DataFrame) -> None:
    """Print a Polars DataFrame using Pandas' wide-table formatting."""
    print(df.to_pandas())


# Step 1: Importing the dataset
wine = pl.read_csv("data/wine_quality_merged.csv")
print(f"Total rows: {len(wine)}")
type_counts = wine["type"].value_counts().to_pandas().set_index("type")["count"]
print(type_counts)

print(" " " ")

# Step 2: Inspecting the data
show(wine.head())

info_pd = pd.DataFrame(
    {
        "Column": wine.columns,
        "Dtype": [str(dtype) for dtype in wine.dtypes],
        "Non-Null Count": [
            len(wine) - wine[col].null_count() for col in wine.columns
        ],
    }
)
print(info_pd.to_string(index=False))

print(" " " ")

# describe() on numeric columns only, oriented like pandas' describe()
# (statistics as columns, data columns as rows)
describe_pd = wine.select(pl.exclude("type")).describe().to_pandas()
print(describe_pd.set_index("statistic").T)

print(" " " ")

print("Missing values:")
null_counts = wine.null_count().to_pandas().iloc[0]
null_counts.name = None
print(null_counts)
print(f"Duplicate rows: {wine.is_duplicated().sum()}")

print(" " " ")

# Step 3: Filtering

# Filtering for high quality wines (quality >= 7)
high_quality = wine.filter(pl.col("quality") >= 7)
print(f"High quality wines: {len(high_quality)} out of {len(wine)}")
show(high_quality[["type", "alcohol", "quality"]].head())

# Filtering for bad quality wines (quality <= 4)
bad_quality = wine.filter(pl.col("quality") <= 4)
print(f"Bad quality wines: {len(bad_quality)} out of {len(wine)}")
show(bad_quality[["type", "alcohol", "quality"]].head())

# Filtering for medium quality wines (4 < quality < 7)
medium_quality = wine.filter((pl.col("quality") > 4) & (pl.col("quality") < 7))
print(f"Medium quality wines: {len(medium_quality)} out of {len(wine)}")
show(medium_quality[["type", "alcohol", "quality"]].head())

# Filtering for red wines with alcohol content greater than 12%
high_alcohol_red = wine.filter((pl.col("type") == "red") & (pl.col("alcohol") > 12))
print(f"High alcohol red wines: {len(high_alcohol_red)} out of {len(wine)}")
show(high_alcohol_red[["type", "alcohol", "quality"]].head())

# Filtering for red wines with alcohol content less than 10%
low_alcohol_red = wine.filter((pl.col("type") == "red") & (pl.col("alcohol") < 10))
print(f"Low alcohol red wines: {len(low_alcohol_red)} out of {len(wine)}")
show(low_alcohol_red[["type", "alcohol", "quality"]].head())

print(" " " ")

# Step 4: Grouping
by_type = wine.group_by("type").agg(
    avg_alcohol=pl.col("alcohol").mean(),
    std_alcohol=pl.col("alcohol").std(),
    min_alcohol=pl.col("alcohol").min(),
    max_alcohol=pl.col("alcohol").max(),
    avg_quality=pl.col("quality").mean(),
    no_of_wines=pl.col("quality").count(),
).sort("type")
show(by_type)

print(" " " ")

by_quality = wine.group_by("quality").agg(
    avg_alcohol=pl.col("alcohol").mean(),
    std_alcohol=pl.col("alcohol").std(),
    min_alcohol=pl.col("alcohol").min(),
    max_alcohol=pl.col("alcohol").max(),
    no_of_wines=pl.col("alcohol").count(),
).sort("quality")
show(by_quality)

print(" " " ")

# Step 5: Machine learning model
print("Machine Learning Model: Predicting Wine Quality")

features = ["alcohol", "volatile acidity", "sulphates"]
X = wine[features].to_numpy()
y = wine["quality"].to_numpy()

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
wine_pd = wine.to_pandas()

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=wine_pd,
    x="quality",
    y="alcohol",
    hue="type",
    palette={"red": "firebrick", "white": "wheat"},
)
plt.title("Alcohol Content by Wine Quality Score")
plt.xlabel("Quality Score")
plt.ylabel("Alcohol (%)")
plt.tight_layout()
plt.savefig("graphs/quality_vs_alcohol_polars.png", dpi=150)
print("Boxplot is saved as quality_vs_alcohol_polars.png")


# Step 7: Visualization of Scatter Plot with Trend Line
plt.figure(figsize=(10, 6))
sns.regplot(
    data=wine_pd,
    x="alcohol",
    y="density",
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"},
)
plt.title("Alcohol Content vs. Density (All Wines)")
plt.xlabel("Alcohol (%)")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("graphs/alcohol_vs_density_polars.png", dpi=150)
print("Scatter plot with trend line saved as alcohol_vs_density_polars.png")
