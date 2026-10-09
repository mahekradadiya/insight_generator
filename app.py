import io
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Healthcare Insight Generator", page_icon="📊", layout="wide")

ID_COLS = ["month", "district"]
DEFAULT_CSV = Path(__file__).parent / "healthcare_data.csv"
SEV_ORDER = ["High", "Medium", "Low"]
SEV_COLORS = {"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2ca02c"}
MIN_RELIABLE_N = 30  # below this many observations a Pearson r is called fragile
INSIGHT_COLS = [
    "insight_id", "type", "indicator", "entity", "period",
    "value", "prev_value", "change_pct", "severity", "explanation",
]
# Cosmetic labels only. Any other column falls back to a title-cased name.
LABEL_OVERRIDES = {"anc_coverage": "ANC Coverage"}


def label(name: str) -> str:
    """Display name for an indicator or an 'a:b' pair."""
    if ":" in name:
        return " vs ".join(label(p) for p in name.split(":"))
    return LABEL_OVERRIDES.get(name, name.replace("_", " ").title())


# ---------------- PART A: DATA LOADING & VALIDATION ----------------

@st.cache_data
def load_data(csv_bytes: bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = [str(c).strip() for c in df.columns]

    missing = set(ID_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["district"] = df["district"].astype(str).str.strip()
    for col in df.columns:
        if col not in ID_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["month"])
    indicators = [c for c in df.columns if c not in ID_COLS and df[c].notna().any()]
    if not indicators:
        raise ValueError("No numeric indicator columns found.")

    df = df[ID_COLS + indicators]
    df = df.dropna(subset=indicators, how="all")
    df = df.sort_values(["district", "month"]).reset_index(drop=True)

    if df.duplicated(["district", "month"]).any():
        raise ValueError("Duplicate district-month rows found. Keep one row per district per month.")
    return df, indicators


# ---------------- PART B: TREND DETECTION ----------------

def compute_trends(data, indicators, threshold):
    """pct_change = (current - previous) / previous * 100 for every (district, indicator)
    pair across consecutive recorded months; is_significant if |pct_change| >= threshold."""
    cols = ["district", "indicator", "prev_month", "month", "previous", "current", "pct_change"]
    rows = []
    for district, group in data.sort_values("month").groupby("district"):
        for ind in indicators:
            s = group[["month", ind]].dropna().reset_index(drop=True)
            for i in range(1, len(s)):
                prev, cur = float(s.at[i - 1, ind]), float(s.at[i, ind])
                pct = np.nan if prev == 0 else (cur - prev) / prev * 100
                rows.append({
                    "district": district, "indicator": ind,
                    "prev_month": s.at[i - 1, "month"], "month": s.at[i, "month"],
                    "previous": prev, "current": cur, "pct_change": pct,
                })
    trends = pd.DataFrame(rows, columns=cols)
    trends["is_significant"] = trends["pct_change"].abs() >= threshold
    return trends


# ---------------- PART C: OUTLIER DETECTION ----------------

def detect_outliers(data, baseline, indicators, method, threshold):
    """Flag rows of `data` whose value is extreme relative to `baseline` statistics.
    `score` is 1.0 exactly on the flagging boundary and grows with distance."""
    cols = ["district", "indicator", "month", "value", "mean", "z", "lower", "upper", "n", "score"]
    rows = []
    for ind in indicators:
        base = baseline[ind].dropna()
        if len(base) < 4:
            continue
        mean, std = base.mean(), base.std(ddof=0)
        q1, q3 = base.quantile(0.25), base.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        for district, month, v in data[["district", "month", ind]].dropna().itertuples(index=False, name=None):
            v = float(v)
            z = (v - mean) / std if std > 0 else 0.0
            if method == "IQR":
                if iqr == 0 or not (v < lower or v > upper):
                    continue
                beyond = (q1 - v) if v < lower else (v - q3)
                score = beyond / (threshold * iqr)
            else:
                if std == 0 or abs(z) < threshold:
                    continue
                score = abs(z) / threshold
            rows.append({
                "district": district, "indicator": ind, "month": month, "value": v,
                "mean": mean, "z": z, "lower": lower, "upper": upper, "n": len(base), "score": score,
            })
    return pd.DataFrame(rows, columns=cols)


# ---------------- PART D: CORRELATION DETECTION ----------------

def detect_correlations(data, indicators, threshold):
    numeric = data[indicators].dropna()
    if len(numeric) < 3 or len(indicators) < 2:
        return pd.DataFrame(), pd.DataFrame(columns=["a", "b", "r", "n"]), len(numeric)
    corr = numeric.corr(method="pearson")
    flagged = []
    for a, b in combinations(indicators, 2):
        r = corr.loc[a, b]
        if pd.notna(r) and abs(r) >= threshold:
            flagged.append({"a": a, "b": b, "r": float(r), "n": len(numeric)})
    return corr, pd.DataFrame(flagged, columns=["a", "b", "r", "n"]), len(numeric)


def limitation_note(n_obs, n_districts, n_months):
    return (
        f"Correlation is based on {n_obs} observations ({n_districts} districts × {n_months} months). "
        "With so few points, Pearson r is fragile: a single extreme district can dominate it, "
        "and it does not establish causation. Recommended minimum: at least 3 months and 10 districts."
    )


# ---------------- PART E: INSIGHT GENERATION ----------------

def severity_from_score(score, medium_x, high_x):
    """score = how many times beyond its own flagging threshold a finding is."""
    if score >= high_x:
        return "High"
    if score >= medium_x:
        return "Medium"
    return "Low"


def build_insights(data, baseline, indicators, trends, outliers, flagged, cfg):
    rows = []

    # Trends
    thr = max(cfg["trend_threshold"], 1e-9)
    for t in trends[trends["is_significant"]].itertuples(index=False):
        gap = (t.month.to_period("M") - t.prev_month.to_period("M")).n
        ref = "the previous month" if gap == 1 else f"the previous recorded month ({t.prev_month:%Y-%m})"
        verb = "increased" if t.pct_change > 0 else "decreased"
        rows.append({
            "type": "trend", "indicator": t.indicator, "entity": t.district,
            "period": t.month.strftime("%Y-%m"),
            "value": round(t.current, 2), "prev_value": round(t.previous, 2),
            "change_pct": round(t.pct_change, 2), "score": abs(t.pct_change) / thr,
            "explanation": (
                f"{label(t.indicator)} in {t.district} {verb} by {abs(t.pct_change):.1f}% compared to {ref} "
                f"({t.previous:g} to {t.current:g}), exceeding the {cfg['trend_threshold']:g}% "
                "significant-change threshold."
            ),
        })

    # Outliers (a different insight type from trends)
    outlier_keys = set()
    for o in outliers.itertuples(index=False):
        outlier_keys.add((o.district, o.indicator, o.month))
        side = "below" if o.value < o.mean else "above"
        if cfg["outlier_method"] == "IQR":
            rule = f"the IQR fence [{o.lower:.1f}, {o.upper:.1f}] (k={cfg['outlier_threshold']:g})"
        else:
            rule = f"the |z| >= {cfg['outlier_threshold']:g} cut-off"
        pct = (o.value - o.mean) / o.mean * 100 if o.mean else np.nan
        rows.append({
            "type": "outlier", "indicator": o.indicator, "entity": o.district,
            "period": o.month.strftime("%Y-%m"),
            "value": round(o.value, 2), "prev_value": round(o.mean, 2),  # prev_value = reference mean
            "change_pct": round(pct, 2) if pd.notna(pct) else np.nan, "score": o.score,
            "explanation": (
                f"{label(o.indicator)} for {o.district} was {o.value:g} in {o.month:%Y-%m}, "
                f"{abs(o.z):.1f}σ {side} the mean ({o.mean:.1f} across {o.n} observations) and outside "
                f"{rule}. Flagging for review."
            ),
        })

    # Correlations: pooled pair + districts where both indicators moved together
    sig = trends[trends["is_significant"]]
    lo_m, hi_m = data["month"].min(), data["month"].max()
    for p in flagged.itertuples(index=False):
        direction = "negative" if p.r < 0 else "positive"
        score = 1 + (abs(p.r) - cfg["corr_threshold"]) / (1 - cfg["corr_threshold"]) if cfg["corr_threshold"] < 1 else 1.0
        fragile = " Based on a small sample, so this estimate is fragile." if p.n < MIN_RELIABLE_N else ""
        rows.append({
            "type": "correlation", "indicator": f"{p.a}:{p.b}", "entity": "All districts",
            "period": f"{lo_m:%Y-%m}..{hi_m:%Y-%m}",
            "value": round(p.r, 3), "prev_value": np.nan, "change_pct": np.nan, "score": score,
            "explanation": (
                f"{label(p.a)} and {label(p.b)} have a {direction} Pearson correlation (r = {p.r:.2f}, n = {p.n}), "
                f"meeting the |r| >= {cfg['corr_threshold']:.2f} threshold.{fragile} "
                "Correlation does not establish causation."
            ),
        })
        both = sig[sig["indicator"] == p.a].merge(
            sig[sig["indicator"] == p.b], on=["district", "month"], suffixes=("_a", "_b"))
        for c in both.itertuples(index=False):
            if np.sign(c.pct_change_a) * np.sign(c.pct_change_b) != np.sign(p.r):
                continue
            rows.append({
                "type": "correlation", "indicator": f"{p.a}:{p.b}", "entity": c.district,
                "period": c.month.strftime("%Y-%m"),
                "value": round(p.r, 3), "prev_value": np.nan, "change_pct": np.nan, "score": score,
                "explanation": (
                    f"In {c.district} for {c.month:%Y-%m}, {label(p.a)} changed {c.pct_change_a:+.1f}% and "
                    f"{label(p.b)} changed {c.pct_change_b:+.1f}%, moving together in line with the overall "
                    f"correlation (r = {p.r:.2f})."
                ),
            })

    # Configurable minimum-coverage breaches (skip values already reported as outliers)
    for ind in cfg["floor_indicators"]:
        if ind not in indicators:
            continue
        std = baseline[ind].std(ddof=0)
        for _, row in data[data[ind] < cfg["coverage_floor"]].iterrows():
            if (row["district"], ind, row["month"]) in outlier_keys:
                continue
            v = float(row[ind])
            score = 1 + (cfg["coverage_floor"] - v) / std if std > 0 else 1.0
            rows.append({
                "type": "threshold_breach", "indicator": ind, "entity": row["district"],
                "period": row["month"].strftime("%Y-%m"),
                "value": round(v, 2), "prev_value": float(cfg["coverage_floor"]), "change_pct": np.nan,
                "score": score,
                "explanation": (
                    f"{label(ind)} in {row['district']} was {v:g}% in {row['month']:%Y-%m}, below the "
                    f"configured minimum of {cfg['coverage_floor']:g}%."
                ),
            })

    if not rows:
        return pd.DataFrame(columns=INSIGHT_COLS)
    out = pd.DataFrame(rows)
    out["severity"] = out["score"].map(lambda s: severity_from_score(s, cfg["medium_x"], cfg["high_x"]))
    out["_rank"] = out["severity"].map({s: i for i, s in enumerate(SEV_ORDER)})
    out = out.sort_values(["_rank", "score"], ascending=[True, False], kind="stable").reset_index(drop=True)
    out["insight_id"] = [f"INS-{i + 1:04d}" for i in range(len(out))]
    return out[INSIGHT_COLS]


def percent_like(series):
    s = series.dropna()
    return len(s) > 0 and s.between(0, 100).all() and s.median() >= 50


def style_severity(df):
    fn = lambda v: f"background-color: {SEV_COLORS.get(v, 'white')}; color: white; font-weight: 600"
    styler = df.style
    return (styler.map if hasattr(styler, "map") else styler.applymap)(fn, subset=["severity"])


# ---------------- PAGE HEADER ----------------

st.title("📊 Automated Healthcare Insight Generation")
st.caption("Explore district performance, identify unusual changes, and generate data-driven insights.")
st.sidebar.header("⚙️ Dashboard Filters")

# ---------------- LOAD DATASET ----------------

try:
    raw = DEFAULT_CSV.read_bytes()
    df, INDICATORS = load_data(raw)
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

# ---------------- PART A: VALIDATION ----------------

with st.expander("Data validation and preview"):
    st.write("**head()**")
    st.dataframe(df.head(), width="stretch")
    buf = io.StringIO()
    df.info(buf=buf)
    st.write("**info()**")
    st.code(buf.getvalue())
    st.write("**Missing values per column**")
    st.dataframe(df.isnull().sum().rename("missing_count").to_frame(), width="stretch")

n_all_districts, n_all_months = df["district"].nunique(), df["month"].nunique()
if n_all_months < 3:
    st.warning(f"Only {n_all_months} month(s) of data; at least 3 are recommended for trend detection.")
if n_all_districts < 10:
    st.warning(f"Only {n_all_districts} districts; at least 10 are recommended for stable correlations.")

# ---------------- LIVE FILTERS ----------------

district_options = sorted(df["district"].unique().tolist())
selected_districts = st.sidebar.multiselect("District", district_options, default=district_options)
selected_indicators = st.sidebar.multiselect("Indicator", INDICATORS, default=INDICATORS, format_func=label)

min_month, max_month = df["month"].min().date(), df["month"].max().date()
date_range = st.sidebar.date_input("Month range", value=(min_month, max_month),
                                   min_value=min_month, max_value=max_month)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_month, max_month

st.sidebar.divider()
st.sidebar.subheader("Detection thresholds")

trend_threshold = st.sidebar.slider("Significant monthly change (%)", 1.0, 100.0, 10.0, 0.5)
outlier_method = st.sidebar.selectbox("Outlier method", ["IQR", "Z-score"])
if outlier_method == "IQR":
    outlier_threshold = st.sidebar.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1)
else:
    outlier_threshold = st.sidebar.slider("Absolute Z-score threshold", 1.0, 5.0, 3.0, 0.1)
full_baseline = st.sidebar.checkbox(
    "Outlier baseline = full dataset", value=True,
    help="Keeps outlier statistics stable when you filter to a few districts or months.")
correlation_threshold = st.sidebar.slider("Absolute Pearson correlation", 0.1, 0.99, 0.7, 0.01)

default_floor_inds = [i for i in INDICATORS if percent_like(df[i])]
floor_indicators = st.sidebar.multiselect(
    "Apply coverage minimum to", INDICATORS, default=default_floor_inds, format_func=label,
    help="Defaults to indicators that look like percentages (all values 0-100, median >= 50).")
coverage_floor = st.sidebar.slider("Minimum acceptable coverage (%)", 0, 100, 75, 5)

with st.sidebar.expander("Severity cut-offs"):
    st.caption("Severity score = how many times beyond its own flagging threshold a finding is.")
    medium_x = st.slider("Medium from (× threshold)", 1.0, 3.0, 1.25, 0.05)
    high_x = st.slider("High from (× threshold)", 1.0, 5.0, 1.75, 0.05)
    if high_x < medium_x:
        st.warning("High cut-off is below Medium; using Medium's value.")
        high_x = medium_x

filtered = df[
    df["district"].isin(selected_districts) & df["month"].dt.date.between(start_date, end_date)
].copy()
active_indicators = [c for c in selected_indicators if c in filtered.columns]

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()
if not active_indicators:
    st.warning("Select at least one indicator in the sidebar.")
    st.stop()

# ---------------- KEY METRICS ----------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(filtered):,}")
c2.metric("Districts", filtered["district"].nunique())
c3.metric("Months", filtered["month"].dt.to_period("M").nunique())
c4.metric("Missing values", int(filtered[active_indicators].isna().sum().sum()))
st.divider()

# ---------------- RUN DETECTORS ----------------

cfg = {
    "trend_threshold": trend_threshold, "outlier_method": outlier_method,
    "outlier_threshold": outlier_threshold, "corr_threshold": correlation_threshold,
    "coverage_floor": coverage_floor, "floor_indicators": floor_indicators,
    "medium_x": medium_x, "high_x": high_x,
}
baseline = df if full_baseline else filtered

trends = compute_trends(filtered, active_indicators, trend_threshold)
outliers = detect_outliers(filtered, baseline, active_indicators, outlier_method, outlier_threshold)
corr, flagged_pairs, n_corr_obs = detect_correlations(filtered, active_indicators, correlation_threshold)
insights = build_insights(filtered, baseline, active_indicators, trends, outliers, flagged_pairs, cfg)

# ---------------- PART E/F: INSIGHT LIST ----------------

st.subheader("🧠 Automatically Generated Insights")

if insights.empty:
    shown = insights
    st.success("No trends, outliers, correlations or threshold breaches were detected "
               "with the current filters and thresholds.")
else:
    types = sorted(insights["type"].unique().tolist())
    chosen_types = st.multiselect("Filter insight types", types, default=types)
    shown = insights[insights["type"].isin(chosen_types)]

    h, m, l = st.columns(3)
    h.metric("High severity", int((shown["severity"] == "High").sum()))
    m.metric("Medium severity", int((shown["severity"] == "Medium").sum()))
    l.metric("Low severity", int((shown["severity"] == "Low").sum()))

    st.dataframe(style_severity(shown), width="stretch", hide_index=True,
                 column_config={"explanation": st.column_config.TextColumn("explanation", width="large")})
    st.download_button("⬇️ Download insights CSV", shown.to_csv(index=False, na_rep="n/a").encode("utf-8"),
                       "healthcare_insights.csv", "text/csv")

st.divider()

# ---------------- PARTS B, C, D: DETECTION DETAILS ----------------

st.subheader("🔎 Detection Details")
tab_trend, tab_out, tab_corr = st.tabs(["Flagged trends", "Outliers", "Correlations"])

with tab_trend:
    flagged_trends = trends[trends["is_significant"]].copy()
    st.write(f"**{len(flagged_trends)}** of {len(trends)} (district, indicator, month) changes meet "
             f"|pct_change| ≥ {trend_threshold:g}%.")
    view = flagged_trends.assign(
        indicator=flagged_trends["indicator"].map(label),
        prev_month=flagged_trends["prev_month"].dt.strftime("%Y-%m"),
        month=flagged_trends["month"].dt.strftime("%Y-%m"),
    ).round({"pct_change": 2})
    st.dataframe(view, width="stretch", hide_index=True)
    with st.expander("All computed changes (including non-significant)"):
        st.dataframe(trends.assign(
            prev_month=trends["prev_month"].dt.strftime("%Y-%m"),
            month=trends["month"].dt.strftime("%Y-%m")).round({"pct_change": 2}),
            width="stretch", hide_index=True)
    st.download_button("⬇️ Download flagged trends",
                       view.to_csv(index=False).encode("utf-8"), "flagged_trends.csv", "text/csv")

with tab_out:
    st.write(f"**{len(outliers)}** outlier(s) via {outlier_method} "
             f"({'k' if outlier_method == 'IQR' else 'z'} = {outlier_threshold:g}).")
    if outliers.empty:
        st.info("No outliers detected.")
    else:
        ov = outliers[["district", "indicator", "month", "value", "z"]].assign(
            indicator=outliers["indicator"].map(label),
            month=outliers["month"].dt.strftime("%Y-%m")).round({"z": 2})
        st.dataframe(ov, width="stretch", hide_index=True)
        st.download_button("⬇️ Download outliers", ov.to_csv(index=False).encode("utf-8"),
                           "outliers.csv", "text/csv")

with tab_corr:
    n_d, n_m = filtered["district"].nunique(), filtered["month"].nunique()
    note = limitation_note(n_corr_obs, n_d, n_m)
    (st.warning if n_corr_obs < MIN_RELIABLE_N else st.caption)(note)
    if corr.empty:
        st.info("Select at least two indicators and ensure at least three complete observations.")
    else:
        st.write("**Pearson correlation matrix**")
        st.dataframe(corr.round(3), width="stretch")
        st.download_button("⬇️ Download correlation matrix", corr.to_csv().encode("utf-8"),
                           "correlation_matrix.csv", "text/csv")
        st.write(f"**Flagged pairs** (|r| ≥ {correlation_threshold:.2f}): {len(flagged_pairs)}")
        if not flagged_pairs.empty:
            st.dataframe(flagged_pairs.assign(a=flagged_pairs["a"].map(label), b=flagged_pairs["b"].map(label))
                         .round({"r": 3}), width="stretch", hide_index=True)

st.divider()

# ---------------- SEVERITY CHART ----------------

st.subheader("📊 Severity Distribution")
if not shown.empty:
    severity_counts = (shown["severity"].value_counts().reindex(SEV_ORDER, fill_value=0)
                       .rename_axis("Severity").reset_index(name="Count"))
    fig_severity = px.bar(severity_counts, x="Severity", y="Count", color="Severity", text="Count",
                          color_discrete_map=SEV_COLORS, category_orders={"Severity": SEV_ORDER},
                          title="Insights by severity")
    fig_severity.update_layout(showlegend=False)
    st.plotly_chart(fig_severity, width="stretch")
else:
    st.info("Severity chart will appear when insights are detected.")

# ---------------- DISTRICT LINE CHART ----------------

st.subheader("📈 Indicator Trend by District")
line_indicator = st.selectbox("Choose indicator for line chart", active_indicators, format_func=label)
line_data = filtered[["month", "district", line_indicator]].dropna().sort_values("month")
fig_line = px.line(line_data, x="month", y=line_indicator, color="district", markers=True,
                   title=f"{label(line_indicator)} over time",
                   labels={"month": "Month", line_indicator: label(line_indicator), "district": "District"})
st.plotly_chart(fig_line, width="stretch")

# ---------------- CORRELATION HEATMAP ----------------

st.subheader("🔥 Pearson Correlation Heatmap")
if not corr.empty:
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr.values, x=[label(x) for x in corr.columns], y=[label(x) for x in corr.index],
        zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
    ))
    fig_heatmap.update_layout(title="Correlation between selected indicators", height=500)
    st.plotly_chart(fig_heatmap, width="stretch")
    st.caption(f"Pairs with |r| ≥ {correlation_threshold:.2f} are flagged. " + limitation_note(
        n_corr_obs, filtered["district"].nunique(), filtered["month"].nunique()))
else:
    st.info("Select at least two indicators and ensure at least three complete observations "
            "are available for the heatmap.")

# ---------------- FILTERED DATA ----------------

st.subheader("📋 Filtered Healthcare Data")
st.dataframe(filtered.sort_values(["month", "district"]), width="stretch", hide_index=True)
st.download_button("⬇️ Download filtered data", filtered.to_csv(index=False).encode("utf-8"),
                   "filtered_healthcare_data.csv", "text/csv")

st.caption("Note: if you use simulated data, insights demonstrate the analytics workflow and should not be "
           "interpreted as verified public-health findings.")