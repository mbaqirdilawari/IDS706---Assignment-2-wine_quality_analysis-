# Wine Quality Analysis (Series 1 of 3)

## About this project

This is Series 1 of a 3-week data science project for IDS 706 (Data Engineering). The
goal this week is to practice the fundamentals: loading a real dataset, inspecting it,
filtering and grouping it, training a first machine learning model, and visualizing it.

## The dataset

**Wine Quality (Red and White)**, sourced from Kaggle:
https://www.kaggle.com/datasets/amirmohamadrezaie/red-and-white-wine-quality

The file `data/wine_quality_merged.csv` contains both red and white wine samples
combined into a single file, with a `type` column that already labels each row as
`"red"` or `"white"`. Each row is one wine sample, described by 11 chemical
measurements, plus a `quality` score from 0–10 assigned by wine tasters. 
The columns are:

`fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur
dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, quality, type`

## How to run this

1. Create a virtual environment and activate it, then install the requirements:
   `pip install pandas matplotlib seaborn scikit-learn`
2. Make sure `wine_quality_merged.csv` is inside a `data/` folder next to `analysis.py`.
3. Run: `python analysis.py`
4. The script prints its findings to the terminal and saves two charts,
   `graphs/quality_vs_alcohol.png` and `graphs/alcohol_vs_density.png`.

---

## Step-by-step walkthrough

### Step 1: Importing the dataset

**What this step does:** 
Loads the single merged CSV file into a pandas DataFrame (a table) and takes a first look at how the two wine types are represented in it.

```python
wine = pd.read_csv("data/wine_quality_merged.csv")
print(f"Total rows: {len(wine)}")
print(wine["type"].value_counts())
```

**Line by line:**
- `wine = pd.read_csv("data/wine_quality_merged.csv")` — reads the CSV file from disk
  and loads it into a DataFrame called `wine`. From this point on, `wine` is the table
  every later step works from.
- `print(f"Total rows: {len(wine)}")` — `len(wine)` counts how many rows the table has;
  the f-string inserts that count directly into the printed sentence.
- `print(wine["type"].value_counts())` — `wine["type"]` pulls out just the `type`
  column; `.value_counts()` counts how many rows fall into each category (`"red"` vs
  `"white"`).

**What we found:** *`[FILL IN: total row count, and the red/white split from
value_counts()]`*

---

### Step 2: Inspecting the data

**What this step does:** Looks at the data's structure and health before doing any
real analysis — what the columns look like, what type of data each holds, and whether
anything is missing or duplicated.

```python
print(wine.head())
print(wine.info())
print(wine.describe())
print("Missing values:")
print(wine.isnull().sum())
print(f"Duplicate rows: {wine.duplicated().sum()}")
```

**Line by line:**
- `print(wine.head())` — shows the first 5 rows, for a quick sanity check.
- `print(wine.info())` — `.info()` prints a summary of every column (name, non-null
  count, data type) by itself; wrapping it in `print()` additionally prints the word
  `None` right after, since `.info()` returns nothing — that stray `None` is harmless
  and can be ignored (or avoided by calling `wine.info()` on its own line without
  `print()`).
- `print(wine.describe())` — for every numeric column, computes count, mean, standard
  deviation, min, the 25th/50th/75th percentiles, and max.
- `print("Missing values:")` — a plain label so the next line's output is easy to read.
- `print(wine.isnull().sum())` — `.isnull()` marks every blank cell as `True`;
  `.sum()` adds those up per column, giving a missing-value count for each one.
- `print(f"Duplicate rows: {wine.duplicated().sum()}")` — `.duplicated()` flags rows
  that are exact copies of an earlier row; `.sum()` counts how many.

**What we found:** *`[FILL IN: anything notable from .describe(), how many missing
values (if any) and how many duplicate rows]`*

---

### Step 3: Filtering

**What this step does:** Pulls out five different meaningful subsets of the data,
using `.query()` to write each condition as plain text. This shows filtering on a
single numeric range, a combined range, and combined conditions across two columns
(type *and* alcohol) at once.

```python
high_quality = wine.query("quality >= 7")
bad_quality = wine.query("quality <= 4")
medium_quality = wine.query("quality > 4 and quality < 7")
high_alcohol_red = wine.query("type == 'red' and alcohol > 12")
low_alcohol_red = wine.query("type == 'red' and alcohol < 10")
```

**Line by line:**
- `high_quality = wine.query("quality >= 7")` — keeps only rows where `quality` is 7
  or higher: the "good" wines.
- `bad_quality = wine.query("quality <= 4")` — keeps only rows where `quality` is 4 or
  lower: the "bad" wines.
- `medium_quality = wine.query("quality > 4 and quality < 7")` — keeps rows strictly
  between 4 and 7, i.e. quality scores of 5 or 6: the "average" wines. The `and`
  combines two conditions into one filter.
- `high_alcohol_red = wine.query("type == 'red' and alcohol > 12")` — combines a
  condition on one column (`type`) with a condition on a different column (`alcohol`)
  in the same filter, keeping only red wines above 12% alcohol.
- `low_alcohol_red = wine.query("type == 'red' and alcohol < 10")` — same idea, red
  wines below 10% alcohol, for comparison against the high-alcohol group above.

Each filtered result is followed by a `print(f"... {len(...)} out of {len(wine)}")`
line (counting how many rows passed) and a `.head()` preview of the `type`, `alcohol`,
and `quality` columns for those rows.

**What we found:** *`[FILL IN: how many wines fell into each of the 5 filtered groups,
out of the total]`*

---

### Step 4: Grouping

**What this step does:** Splits the dataset into groups and computes summary
statistics for each group — first by wine type, then by quality score.

```python
by_type = wine.groupby("type").agg(
    avg_alcohol=("alcohol", "mean"),
    avg_quality=("quality", "mean"),
    no_of_wines=("quality", "count"),
)

by_quality = wine.groupby("quality").agg(
    avg_alcohol=("alcohol", "mean"),
    no_of_wines=("alcohol", "count"),
)
```

**Line by line:**
- `wine.groupby("type")` — splits the table into two piles: all `"red"` rows and all
  `"white"` rows.
- `.agg(avg_alcohol=("alcohol", "mean"), avg_quality=("quality", "mean"),
  no_of_wines=("quality", "count"))` — for each pile, computes three things at once and
  names each result column: the average alcohol (`avg_alcohol`), the average quality
  (`avg_quality`), and how many rows are in that pile (`no_of_wines`, computed by
  counting the non-blank `quality` values — since every row has one, this is just the
  row count for that group).
- `wine.groupby("quality")` — same idea, but the piles are formed by quality score
  (all the 3s, all the 4s, and so on) instead of by type.
- `.agg(avg_alcohol=("alcohol", "mean"), no_of_wines=("alcohol", "count"))` — for each
  quality score, the average alcohol content and how many wines got that score.

**What we found:** *`[FILL IN: the avg_alcohol/avg_quality/no_of_wines numbers for red
vs. white, and whether alcohol trends up or down as quality score increases]`*

---

### Step 5: Machine learning model

**What this step does:** Trains a Linear Regression model — the simplest predictive
model there is — to predict a wine's quality score from three of its chemical
properties, then measures how good its predictions were on data it never saw during
training.

```python
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
```

**Line by line:**
- `features = ["alcohol", "volatile acidity", "sulphates"]` — the three columns chosen
  as inputs to the model.
- `X = wine[features]` — the inputs table (capital `X` is the standard name for
  "inputs" in machine learning code).
- `y = wine["quality"]` — the correct answers (lowercase `y` is the standard name for
  "target").
- `train_test_split(X, y, test_size=0.2, random_state=42)` — randomly splits the rows
  into 80% for training (`X_train`, `y_train`) and 20% for testing (`X_test`,
  `y_test`). `random_state=42` makes the split reproducible — running this again gives
  the exact same split rather than a new random one.
- `model = LinearRegression()` — creates a blank, untrained model.
- `model.fit(X_train, y_train)` — the learning step: the model works out the
  mathematical relationship between the training inputs and their correct answers.
- `predictions = model.predict(X_test)` — asks the trained model to guess the quality
  of the test wines, which it has never seen.
- `mean_squared_error(y_test, predictions)` — measures how far off the guesses were
  from the real answers, on average (squared, so bigger misses count more). Lower is
  better.
- `r2_score(y_test, predictions)` — measures what fraction of the variation in quality
  the model explains, from 0 (no better than guessing the average every time) to 1
  (perfect).

Each feature's learned coefficient is then printed with a loop:
```python
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.3f}")
```
`model.coef_` holds one number per feature, in the same order as `features`; `zip()`
pairs each feature name with its coefficient so the loop can print both together. A
positive coefficient means "as this feature goes up, predicted quality tends to go up
too"; negative means the opposite.

**What we found:** *`[FILL IN: the Mean Squared Error, R-squared, and the three
coefficients — plus a sentence on which feature had the strongest effect and in which
direction]`*

---

### Step 6: Visualization — boxplot

**What this step does:** Draws a boxplot comparing alcohol content across quality
scores, split by wine type.

```python
sns.boxplot(data=wine, x="quality", y="alcohol", hue="type")
```

**Why alcohol and quality:** Alcohol is one of the three features the model above uses
to predict quality, so this chart lets you *see* that relationship directly instead of
just reading a coefficient. `hue="type"` adds a second comparison for free — red vs.
white — on the same chart.

**Line by line:**
- `plt.figure(figsize=(10, 6))` — starts a blank chart canvas, 10 by 6 inches.
- `sns.boxplot(data=wine, x="quality", y="alcohol", hue="type")` — one box per quality
  score on the x-axis, showing the spread of alcohol values (y-axis) for wines with
  that score; `hue="type"` splits each box into a red-wine box and a white-wine box
  side by side.
- `plt.title(...)`, `plt.xlabel(...)`, `plt.ylabel(...)` — label the chart and axes.
- `plt.tight_layout()` — adjusts spacing so nothing overlaps or gets cut off.
- `plt.savefig("quality_vs_alcohol.png", dpi=150)` — saves the chart as an image file.

**What we found:** *`[FILL IN: describe the pattern — does alcohol trend up or down as
quality increases, and is the pattern different for red vs. white?]`*

---

### Step 7: Visualization — scatter plot

**What this step does:** Adds a second chart, this time comparing two continuous
chemical properties directly against each other rather than against the discrete
quality score.

```python
plt.figure(figsize=(10, 6))
sns.regplot(data=wine, x="alcohol", y="density", scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
plt.title("Alcohol Content vs. Density (All Wines)")
plt.xlabel("Alcohol (%)")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("alcohol_vs_density.png", dpi=150)
print("Scatter plot with trend line saved as alcohol_vs_density.png")
```

**Why alcohol and density (and not quality again):** `quality` only takes whole
numbers from 3 to 9, so a scatter plot with `quality` on an axis produces vertical
stripes of points rather than a smooth cloud — a boxplot (Step 6) is the better tool
for that comparison. Alcohol and density are both continuous measurements, and wine
chemistry gives a real reason to expect a relationship between them: alcohol is less
dense than water, so wines with more alcohol tend to have lower density. That makes
this pair a clearer, more classic example of what a scatter plot is for. This version
also combines red and white into one group rather than splitting by `hue="type"`, to
show the overall trend across every wine at once.

**Line by line:**
- `plt.figure(figsize=(10, 6))` — new blank chart canvas.
- `sns.regplot(data=wine, x="alcohol", y="density", scatter_kws={"alpha": 0.3},
  line_kws={"color": "red"})` — plots one point per wine (alcohol on the x-axis,
  density on the y-axis) and additionally fits a straight trend line through all of
  them using linear regression under the hood — the same core idea as Step 5's model,
  just applied to two variables and drawn directly on the chart instead of printed as
  numbers. `scatter_kws={"alpha": 0.3}` makes the dots partly transparent (so
  overlapping points, of which there are thousands, show up as darker patches instead
  of a solid blob); `line_kws={"color": "red"}` makes the trend line red so it stands
  out against the dots. `regplot` also draws a shaded band around the line showing its
  uncertainty.
- `plt.title(...)`, `plt.xlabel(...)`, `plt.ylabel(...)` — label the chart and axes.
- `plt.tight_layout()` — fixes spacing.
- `plt.savefig("alcohol_vs_density.png", dpi=150)` — saves the chart as an image file.
- `print(...)` — confirms the file was saved.

**What we found:** *`[FILL IN: describe the pattern — does the trend line slope
downward (density decreasing as alcohol increases), as expected from wine chemistry?
How tight or scattered are the points around the line?]`*

---

## Overall findings

*`[FILL IN once the numbers above are in: 2-4 sentences tying it together — e.g. which
wine type tends to score higher, what predicts quality best, and whether a 3-feature
linear model is good enough or too simple.]`*

## Next steps (for later weeks)

- Try more features in the regression model, or a different algorithm (e.g., Random
  Forest).
- Compare Pandas performance against Polars on this same dataset.
- Add tests and set up continuous integration (CI) for this script.
