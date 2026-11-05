# Data Mining Agent – Indian Election Dataset

This ZIP contains a Python Streamlit application configured with your uploaded `dataset2.csv`.

## Included algorithms
1. Apriori association-rule mining
2. FP-Growth association-rule mining
3. ID3 decision tree using information gain
4. C4.5 decision tree using gain ratio
5. K-Means clustering
6. Agglomerative clustering
7. DBSCAN clustering

The app automatically loads `dataset2.csv` when opened. You can optionally upload a different CSV from the app.

## Run in VS Code (Windows)

1. Extract this ZIP file.
2. Open the extracted folder in VS Code.
3. Open **Terminal > New Terminal**.
4. Create a virtual environment:

```powershell
py -m venv .venv
```

5. Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

6. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

7. Start the application:

```bash
streamlit run app.py
```

The app opens in your browser. Keep the terminal running while using it.

## Dataset details

`dataset2.csv` contains 178 rows and these 7 columns:
- Year
- State
- Total_Seats
- Male_Candidate
- Female_Candidate
- Male_Winner
- Female_Winner

There are no missing cells in the uploaded dataset.

## How each algorithm uses this dataset

### Apriori and FP-Growth
Each state-year row is converted into a transaction of categorical items. Numeric columns are converted into Low / Medium / High levels (where applicable). The app then mines frequent itemsets and association rules. Support and confidence are adjustable.

### ID3 and C4.5
The app creates Low / Medium / High category columns for numeric columns with more than three unique values. These engineered columns can be selected as a classification target. ID3 uses information gain; C4.5 uses gain ratio. Numeric input features are split using thresholds.

### Clustering
Choose numeric election-statistics columns. Features are standardised before K-Means, Agglomerative, or DBSCAN is applied. The app displays cluster labels, cluster counts, a PCA plot, and a downloadable result.

## Notes
- For meaningful decision-tree classification, use a categorical or engineered level target such as `Female_Winner_Level` rather than a raw count with many unique values.
- Apriori / FP-Growth results depend on selected columns and minimum support. They show associations, not cause-and-effect.
- The dataset is included in the ZIP so the app can run without uploading it again.
