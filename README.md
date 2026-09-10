# IDS706---Assignment-1

# Wine Quality Analysis

## About this Project

This is Series 1 of a 3-week data science project for IDS 706 (Data Engineering). The
goal is to practice the fundamentals: loading a real dataset, inspecting it,
filtering and grouping it, training a first machine learning model, and visualizing it.

## The Dataset

**Wine Quality (Red and White)**, sourced from Kaggle:
https://www.kaggle.com/datasets/amirmohamadrezaie/red-and-white-wine-quality

The file `data/wine_quality_merged.csv` contains both red and white wine samples
combined into a single file, with a `type` column that already labels each row as
`"red"` or `"white"`. Each row is one wine sample, described by 11 chemical
measurements, plus a `quality` score from 0–10 assigned by wine tasters. 
The columns are:
'fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur
dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, quality, type`

## How to run this

These steps assume you've already cloned this repository and have a terminal open inside the `wine-quality-analysis` folder.

### 1. Create a virtual environment

Run this in the **terminal**:

```bash
python3 -m venv .venv
```

This creates a folder called `.venv` that holds a clean, isolated copy of Python just for this project, so the packages you install don't clash with anything else on your machine.

### 2. Activate the virtual environment

Run this in the **terminal**:

```bash
source .venv/bin/activate
```

You'll know it worked because your terminal prompt will now show `(.venv)` at the start of the line. You need to run this activation command every time you open a new terminal window to work on this project.

### 3. Install the required packages

Run this in the **terminal** (with the virtual environment still active):

```bash
pip install -r requirements.txt
```

This reads the `requirements.txt` file in this repo and installs the four libraries the script needs: `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.

### 4. Add the dataset

This is a **file/folder step, not a terminal command**: 
Make sure `wine_quality_merged.csv` is placed inside a folder named `data/`, sitting right next to `analysis.py`. The folder structure should look like this:

wine-quality-analysis/
├── analysis.py
├── data/
│ └── wine_quality_merged.csv
├── graphs/
├── requirements.txt
└── README.md


If you don't have the dataset yet, download it from the Kaggle link in the "Dataset" section above and place it in the `data/` folder.

### 5. Run the script

Run this in the **terminal**:

```bash
python analysis.py
```

This runs every step of `analysis.py` from top to bottom: it loads the dataset, prints inspection details, prints the filtering and grouping results, trains the model and prints its performance, and finally saves two charts.

### 6. Check the output

You don't need to run anything for this step — just look at what happened:

- All the printed results (row counts, `.describe()` output, filter counts, group tables, model error and R-squared) appear directly in your **terminal**.
- Two image **files** are created inside the `graphs/` folder: `graphs/quality_vs_alcohol.png` and `graphs/alcohol_vs_density.png`. Open these from VS Code's file explorer (or any image viewer) to see the charts.

### Optional: using the Makefile shortcuts

If you'd rather not type each command separately, this repo includes a `Makefile` with shortcuts. These also run in the **terminal**:

- `make setup` — creates the virtual environment and installs the requirements (does steps 1–3 for you)
- `make run` — runs the script (does step 5 for you)
- `make clean` — deletes the generated charts and cached Python files, useful if you want a fresh run

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

**What we found:** 
*`The dataset has 6,497 wines total.`*
*`4,898 white (about 75%) and 1,599 red (about 25%), so white wines make up the large majority of the combined file.`*

---

### Step 2: Inspecting the Data

**What this step does:** 
Looks at the data's structure and health before doing any
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

**Line by Line:**
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

**What we found:** 
*No column has any missing values. Every one of the 13 columns shows 0 missing across all 6,497 rows.*
*There are, however, 1,177 exact duplicate rows (about 18% of the dataset), rows that repeat another row's values identically.* 
*.describe() shows that most chemical measurements are fairly tight (e.g. alcohol ranges from 8.0% to 14.9%, averaging 10.49%), but residual sugar is heavily skewed: its 75th percentile is 8.1 but its maximum is 65.8, meaning a small number of unusually sweet wines pull the average up.*

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

**Line by Line:**
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

**What we found:** 
If we consider alcohol quality, out of 6,497 wines: 
- 1,277 (about 20%) are high quality (score ≥ 7)
- 246 (about 4%) are bad quality (score ≤ 4)
- the remaining 4,974 (about 77%) fall in the medium range
*So most wines cluster around average quality, and truly bad wines are rare.* 

Secondly, among red wines specifically:
- 141 (about 9% of all reds) have alcohol above 12%
- 680 (about 43% of all reds) have alcohol below 10%
*Therefore, lower-alcohol reds are considerably more common than higher-alcohol ones.*

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

**What we found:** 
*White wines average a slightly higher quality score than red (5.88 vs. 5.64) despite very similar average alcohol content (10.51% vs. 10.42%).*
*White also shows more variation in alcohol (std 1.23 vs. 1.07).* 
*The alcohol-by-quality relationship isn't perfectly straight-line: quality scores 3 and 4 actually have slightly higher average alcohol (10.2%) than quality 5 (9.8%, the lowest point in the table), but from quality 5 upward the trend climbs steadily and clearly — 9.8% → 10.6% → 11.4% → 11.7% → 12.2% across quality 5 through 9. Overall, higher-quality wines do tend to have more alcohol, especially in the upper half of the range.*

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

**What we found:** 
*The model's Mean Squared Error was 0.551 and R-squared was 0.253*
*These three features (alcohol, volatile acidity, sulphates) explain about 25% of the variation in quality, a real but modest amount, since quality clearly depends on more than three chemical measurements.* 

- *Volatile acidity had by far the strongest effect (coefficient −1.466): as it increases, predicted quality drops sharply, consistent with volatile acidity being linked to a vinegar-like taste.* 
- *Sulphates (+0.641) and alcohol (+0.322) both had smaller, positive effects. More of either tends to predict slightly higher quality.*

---

### Step 6: Visualization — Boxplot

**What this step does:** 
*Draws a boxplot comparing alcohol content across quality scores, split by wine type.*

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

**What we found:** 
*The boxplot's pattern matches the by_quality numbers above: alcohol content generally climbs as quality score increases, most clearly from quality 5 onward, and the trend looks broadly similar for red and white, though white's boxes show a bit more spread at the low end.*

---

### Step 7: Visualization — Scatter Plot

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
plt.savefig("graphs/alcohol_vs_density.png", dpi=150)
print("Scatter plot with trend line saved as alcohol_vs_density.png")
```

**Why alcohol and density (and not quality again):** 
`quality` only takes whole numbers from 3 to 9, so a scatter plot with `quality` on an axis produces vertical
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

**What we found:** 
*The scatter plot shows a clear inverse relationship.*
*As alcohol content goes up, density tends to go down. The red trend line slopes downward across the whole range, confirming this. Most points are tightly packed in a diagonal band between about 8–14% alcohol and a density of 0.99–1.00, which makes sense chemically: alcohol is less dense than water, so wines with more alcohol are naturally less dense.*
*There are a few outliers worth noting though. One wine near 11.5% alcohol has an unusually high density (about 1.04), and one near 8.8% alcohol sits at about 1.01, both well above the rest of the cloud. Aside from those outliers, the relationship is fairly consistent and fits a straight line reasonably well, though the points do fan out a bit more at the lower end of alcohol content than at the higher end. Check out graphs/alcohol_vs_density.png yourself below and confirm.*


---

## Overall Findings

*`[FILL IN once the numbers above are in: 2-4 sentences tying it together — e.g. which
wine type tends to score higher, what predicts quality best, and whether a 3-feature
linear model is good enough or too simple.]`*

## Next Steps (for later weeks)

- Try more features in the regression model, or a different algorithm (e.g., Random
  Forest).
- Compare Pandas performance against Polars on this same dataset.
- Add tests and set up continuous integration (CI) for this script.
