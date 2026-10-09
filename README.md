# Automated Healthcare Insight Generation

A small dashboard that reads district-level healthcare data and **automatically tells you what changed, what looks unusual, and what moves together**, in plain sentences.

Example of what it writes on its own:

> ANC Coverage in Ahmedabad decreased by 18.8% compared to the previous month (85 to 69), exceeding the 10% significant-change threshold.

Built with Python, Streamlit, pandas and Plotly. No separate backend.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. Keep `healthcare_data.csv` in the same folder as `app.py`.

Optional: create a virtual environment first (`python -m venv .venv`, then activate it).

---

## Files

| File | What it is |
|---|---|
| `app.py` | The whole app: analysis logic and dashboard |
| `healthcare_data.csv` | The dataset (6 districts x 2 months) |
| `requirements.txt` | Libraries to install |
| `sample_insights.json` | The insights the app produces on this dataset with default settings |
| `README.md` | This guide |

---

## The dataset

12 rows: Ahmedabad, Surat, Vadodara, Rajkot, Mehsana and Bhavnagar, for 2026-07 and 2026-08.

| Column | Meaning | Unit |
|---|---|---|
| `month` | Month of the record | YYYY-MM |
| `district` | District name | text |
| `anc_coverage` | Antenatal care coverage | % |
| `institutional_delivery` | Deliveries in health facilities | % |
| `immunization` | Immunization coverage | % |
| `high_risk_cases` | High-risk pregnancy cases | count |

Rules: one row per district per month. Any extra numeric columns are picked up automatically as indicators, so you can use other data without changing the code.

---

## What it finds on this data

With the default settings the app produces **14 insights: 10 High, 2 Medium, 2 Low**. The main ones:

| What | Where | Result |
|---|---|---|
| **Trend** | Ahmedabad ANC coverage | 85 to 69 (-18.8%), High |
| **Trend** | Mehsana ANC coverage | 84 to 42 (-50%), High |
| **Trend** | Mehsana high-risk cases | 11 to 28 (+154.5%), High |
| **Outlier** | Mehsana ANC coverage = 42 | Far below the average of 78.3, High |
| **Outlier** | Mehsana high-risk cases = 28 | Far above the average of 13.4, High |
| **Correlation** | ANC coverage vs high-risk cases | r = -0.93: when ANC falls, high-risk cases rise |
| **Correlation** | Institutional delivery vs immunization | r = 0.98: they rise and fall together |

Reading it together: **Mehsana in August 2026 is the standout.** ANC coverage collapsed while high-risk cases jumped, and the app links the two through the negative correlation. Ahmedabad shows the same pattern in a milder form (ANC down 18.8%, high-risk cases up 30%).

All 14 insights are listed in `sample_insights.json`.

---

## Using the dashboard

**Sidebar**
- **Filters:** district, indicator, month range. Everything updates live.
- **Detection thresholds** (all sliders):
  - Significant monthly change (%): default 10
  - Outlier method: IQR (default) or Z-score, with its own slider
  - Absolute Pearson correlation: default 0.70
  - Minimum acceptable coverage (%): default 75
- **Severity cut-offs:** where Medium and High begin (see below).
- **Outlier baseline = full dataset:** keeps outlier checks fair when you filter down to a few districts.

**Main page, top to bottom**
1. Data validation (`head()`, `info()`, missing values)
2. Key numbers (records, districts, months, missing values)
3. **Insights table** with severity colours, an insight-type filter, and CSV / JSON downloads
4. **Detection details** tabs: flagged trends, outliers, correlations (matrix and flagged pairs)
5. Charts: severity counts (bar), indicator by district (line), correlation heatmap
6. The filtered data, with download

---

## How it works (in plain words)

**Trends.** For each district and indicator, compare each month with the previous one:
`pct_change = (current - previous) / previous x 100`.
If the change is at least the threshold (default 10%), it is flagged as significant.

**Outliers.** Look at all values of an indicator across all districts and months.
- *IQR rule:* a value is an outlier if it is above Q3 + 1.5 x IQR or below Q1 - 1.5 x IQR.
- *Z-score:* a value is an outlier if it is 3 or more standard deviations from the mean.

Trends (change over time) and outliers (unusual value) are kept as separate insight types.

**Correlations.** Calculate the Pearson correlation between every pair of indicators. Pairs with |r| of 0.70 or more are flagged. If two flagged indicators both changed significantly in the same district and month, in the direction the correlation predicts, that district gets its own insight too.

**Threshold breaches.** A coverage value below the minimum-coverage slider is flagged (applies to the percentage indicators).

**Insights.** Each finding becomes one row with these fields:

| Field | Meaning |
|---|---|
| `insight_id` | INS-0001, INS-0002, ... |
| `type` | trend, outlier, correlation, or threshold_breach |
| `indicator` | Which measure |
| `entity` | District (or "All districts") |
| `period` | Month |
| `value` | The number (for correlations, r) |
| `prev_value` | Previous month (trend), average (outlier), or minimum (breach) |
| `change_pct` | Percent change |
| `severity` | Low, Medium or High |
| `explanation` | Auto-written sentence |

The sentences come from templates, but **every number comes from the data**. Nothing is hardcoded for a specific district.

---

## How severity is decided

Each finding gets a score: **how many times beyond its own threshold it is.**

Example: Ahmedabad's ANC change is 18.8% and the threshold is 10%, so the score is 1.88.

- Score 1.75 or more: **High**
- Score 1.25 to 1.75: **Medium**
- Below 1.25: **Low**

Both cut-offs are sliders. Outliers, correlations and breaches use the same idea, measured against their own thresholds.

---

## Important limitations

1. **Correlations are fragile here.** There are only 12 observations (2 months x 6 districts), and Mehsana alone drives much of the pattern. Treat the correlation values as hints, not proof. The app shows a warning with the real count whenever there are fewer than 30 observations. The recommended minimum is 3 months and 10 districts. Correlation also does not prove one thing causes another.
2. **Data-size warnings are expected.** The yellow messages "Only 2 month(s)..." and "Only 6 districts..." appear because this sample is smaller than recommended. They disappear with larger data.
3. **Z-score does not flag Mehsana at the default of 3.** Mehsana's ANC value is just under 3 standard deviations from the mean (it displays as 3.0 but is about 2.95), because the extreme value itself inflates the spread. The IQR rule, which is the default, flags it. Lower the Z-score slider to 2.5 to see it flagged by Z-score.
4. **No threshold-breach insights appear by default.** Only two coverage values are below 75: Mehsana's 42 and Ahmedabad's 69. Both are already reported as outliers, so they are not repeated as breaches. Every other value is 77 or above. Raise the slider to 80 to see breaches for Rajkot and Bhavnagar.
5. **Outliers are compared across all months and districts together.** With more history, per-district baselines would be more precise.
6. **Missing months:** trends compare consecutive recorded months. If a month is missing, the sentence names the actual earlier month used.