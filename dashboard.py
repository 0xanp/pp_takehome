import json
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1) SINGLE-PAGE LOADING & CACHING
# ---------------------------------------------------------

@st.cache_data
def load_data():
    """
    Load parcel_table and log_table from Parquet using pandas,
    convert key columns to timezone-naive datetime.
    Returns:
        parcel_df (pd.DataFrame): Parcel table
        log_df (pd.DataFrame): Log table
    """
    try:
        # --- READ PARQUET FILES ---
        parcel_df = pd.read_parquet("parcel_table.pqt")
        log_df = pd.read_parquet("log_table.pqt")

        # Convert parcel date columns to naive datetime
        date_cols = ["picked_up_date", "out_for_delivery_date", "first_attempt_date", "final_delivery_date"]
        for col in date_cols:
            if col in parcel_df.columns:
                parcel_df[col] = pd.to_datetime(parcel_df[col], errors="coerce").dt.tz_localize(None)

        # Convert log_timestamp to naive datetime
        if "log_timestamp" in log_df.columns:
            log_df["log_timestamp"] = pd.to_datetime(log_df["log_timestamp"], errors="coerce").dt.tz_localize(None)

        return parcel_df, log_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# ---------------------------------------------------------
# 2) PART 1: SQL SNIPPETS + EXPLANATIONS
# ---------------------------------------------------------
def render_part1():
    """
    Display SQL query snippets for Part 1 along with explanations.
    """
    st.markdown("""
## Part 1: SQL Analysis (Conceptual)

Below are the SQL queries (and brief notes) addressing Part 1 questions. These snippets illustrate how to run them against the database—no runtime computation is performed here.

**1. Average, Median, and 90th Percentile Transit Time (Domestic, 2024):**
```sql
SELECT
    AVG(DATEDIFF(final_delivery_date, picked_up_date))       AS avg_days,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY DATEDIFF(final_delivery_date, picked_up_date)
    )                                                       AS median_days,
    PERCENTILE_CONT(0.9) WITHIN GROUP (
        ORDER BY DATEDIFF(final_delivery_date, picked_up_date)
    )                                                       AS p90_days
FROM parcel_table
WHERE origin_country = destination_country
  AND YEAR(picked_up_date) = 2024;
```
- *Explanation*: Filters for domestic shipments in 2024. Computes average, median, and 90th percentile of days between `picked_up_date` and `final_delivery_date`.

**2. Max Transit Time and Parcel Count:**
```sql
SELECT
    DATEDIFF(final_delivery_date, picked_up_date) AS transit_days,
    COUNT(*)                                   AS num_parcels
FROM parcel_table
WHERE is_delivered = TRUE
  AND final_delivery_date IS NOT NULL
  AND picked_up_date IS NOT NULL
GROUP BY transit_days
ORDER BY transit_days DESC
LIMIT 1;
```
- *Explanation*: Finds the maximum transit days and how many parcels share that maximum.

**3. Top 2 Carriers by Volume per Trade Lane:**
```sql
WITH carrier_volume AS (
    SELECT
        origin_country,
        destination_country,
        carrier_name,
        COUNT(*)                       AS parcel_volume,
        ROW_NUMBER() OVER (
            PARTITION BY origin_country, destination_country
            ORDER BY COUNT(*) DESC
        )                             AS rn
    FROM parcel_table
    GROUP BY origin_country, destination_country, carrier_name
)
SELECT
    origin_country,
    destination_country,
    carrier_name,
    parcel_volume
FROM carrier_volume
WHERE rn <= 2
ORDER BY origin_country, destination_country, rn;
```
- *Explanation*: Ranks carriers by parcel count per (origin, destination) pair and selects top 2.

**4. Delivered Parcels without Any Log Records:**
```sql
SELECT p.*
FROM parcel_table p
LEFT JOIN log_table l
  ON p.parcel_id = l.parcel_id
WHERE p.is_delivered = TRUE
  AND l.parcel_id IS NULL;
```
- *Explanation*: Identifies delivered parcels with no associated log entries.

**5. Parcels with Multiple Distinct Carriers:**
```sql
SELECT
    parcel_id,
    STRING_AGG(DISTINCT carrier_name, '; ') AS list_of_carrier
FROM parcel_table
GROUP BY parcel_id
HAVING COUNT(DISTINCT carrier_name) > 1;
```
- *Explanation*: Lists parcels handled by more than one carrier, concatenating distinct carrier names.
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3) PART 2: EDD EXTRACTION & ACCURACY
# ---------------------------------------------------------
def extract_edd_from_log_group(group: pd.DataFrame) -> tuple:
    """
    Parse log rows for a single parcel_id to extract EDD start/end times from additional_params JSON.
    Returns latest (by log_timestamp) pair (start, end) as timezone-naive datetimes, else (None, None).
    """
    edd_records = []
    for _, row in group.iterrows():
        params_json = row.get("additional_params")
        if not isinstance(params_json, str):
            continue
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError:
            continue

        starts = {k: params[k] for k in params if k.endswith("_start") and "expected_time_" in k and params[k]}
        ends = {k: params[k] for k in params if k.endswith("_end") and "expected_time_" in k and params[k]}
        for s_key, s_val in starts.items():
            prefix = s_key[:-6]  # strip '_start'
            e_key = prefix + "_end"
            if e_key in ends:
                try:
                    s_dt = pd.to_datetime(s_val).tz_localize(None)
                    e_dt = pd.to_datetime(ends[e_key]).tz_localize(None)
                    edd_records.append((s_dt, e_dt, row.get("log_timestamp")))
                except Exception:
                    pass
    if not edd_records:
        return None, None
    edd_records.sort(key=lambda x: x[2] if x[2] is not None else pd.Timestamp.min)
    return edd_records[-1][0], edd_records[-1][1]

@st.cache_data
def compute_edd_fields(parcel_df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each parcel, extract log-based EDD and compute edd_mid_log. Then compute edd_final using fallback (pickup + median transit).
    Returns merged DataFrame with edd_mid_log and edd_final.
    """
    edd_list = []
    for pid, group in log_df.groupby("parcel_id"):
        s, e = extract_edd_from_log_group(group)
        if s is not None and e is not None:
            edd_list.append({"parcel_id": pid, "edd_start_log": s, "edd_end_log": e})
    edd_log_df = pd.DataFrame(edd_list)

    merged = parcel_df.merge(edd_log_df, on="parcel_id", how="left")
    merged["edd_mid_log"] = merged.apply(
        lambda r: r["edd_start_log"] + (r["edd_end_log"] - r["edd_start_log"]) / 2
        if pd.notna(r.get("edd_start_log")) and pd.notna(r.get("edd_end_log")) else pd.NaT,
        axis=1,
    )

    # Compute fallback median transit days per carrier (using actual delivered parcels)
    delivered_full = merged.loc[
        (merged["is_delivered"] == True)
        & merged["picked_up_date"].notna()
        & merged["final_delivery_date"].notna()
    ].copy()
    delivered_full["transit_days"] = (
        delivered_full["final_delivery_date"] - delivered_full["picked_up_date"]
    ).dt.days
    median_transit = delivered_full.groupby("carrier_name")["transit_days"].median()

    # Compute edd_final for each row
    def compute_edd_final(row):
        mid = row.get("edd_mid_log")
        if pd.notna(mid):
            return mid
        pu = row.get("picked_up_date")
        if pd.isna(pu):
            return pd.NaT
        carrier = row.get("carrier_name")
        med = median_transit.get(carrier, np.nan)
        if pd.notna(med):
            return pu + timedelta(days=int(med))
        return pd.NaT

    merged["edd_final"] = merged.apply(compute_edd_final, axis=1)
    return merged

# ---------------------------------------------------------
# 3) PART 2: EDD EXTRACTION & ACCURACY
# ---------------------------------------------------------
def render_part2(parcel_with_edd: pd.DataFrame) -> None:
    """
    Render Part 2: EDD extraction and accuracy analysis using edd_final column.
    """
    st.markdown("""
## Part 2: EDD Accuracy (Log + Fallback)

Using the precomputed `edd_final`, measure `|final_delivery_date - edd_final|`.
""", unsafe_allow_html=True)

    edd_df = parcel_with_edd.copy()

    # Filter valid delivered parcels with edd_final
    edd_valid = edd_df.loc[
        edd_df["edd_final"].notna()
        & edd_df["final_delivery_date"].notna()
        & (edd_df["is_delivered"] == True)
    ].copy()
    if edd_valid.empty:
        st.info("No parcels have a computed edd_final and final delivery date.")
        return

    edd_valid["edd_accuracy_days"] = (
        edd_valid["final_delivery_date"] - edd_valid["edd_final"]
    ).dt.days

    # Show raw deviations for UNKNOWN
    unknown_raw = edd_valid[edd_valid["carrier_name"] == "UNKNOWN"]
    if not unknown_raw.empty:
        st.subheader("🔍 Raw EDD Deviations for 'UNKNOWN'")
        st.dataframe(
            unknown_raw[["parcel_id", "carrier_name", "edd_accuracy_days"]].sort_values("edd_accuracy_days"),
            use_container_width=True,
        )

    # Exclude extreme deviations beyond ±365 days (likely data errors)
    edd_valid = edd_valid[edd_valid["edd_accuracy_days"].abs() <= 365]

    # Compute overall metrics
    total_count = len(edd_valid)
    on_time_count = edd_valid[edd_valid["edd_accuracy_days"].abs() <= 1].shape[0]
    overall_accuracy = on_time_count / total_count * 100
    avg_dev = edd_valid["edd_accuracy_days"].mean().round(1)
    coverage = total_count / len(edd_df) * 100

    st.subheader("🎯 EDD Accuracy Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("📊 Overall Accuracy (±1 day)", f"{overall_accuracy:.1f}%")
    with c2:
        st.metric("📈 Avg Deviation", f"{avg_dev} days")
    with c3:
        st.metric("📋 Parcels with EDD", f"{total_count:,}", f"{coverage:.1f}% of total")

    st.markdown("---")
    st.subheader("🚚 Carrier EDD Performance")
    carrier_stats = (
        edd_valid.groupby("carrier_name").agg(
            total_with_edd=pd.NamedAgg(column="parcel_id", aggfunc="count"),
            avg_deviation=pd.NamedAgg(column="edd_accuracy_days", aggfunc="mean"),
            std_deviation=pd.NamedAgg(column="edd_accuracy_days", aggfunc="std"),
            on_time=pd.NamedAgg(column="edd_accuracy_days", aggfunc=lambda x: (x.abs() <= 1).sum()),
        )
        .round(2)
    )
    carrier_stats["accuracy_rate"] = (carrier_stats["on_time"] / carrier_stats["total_with_edd"] * 100).round(1)
    carrier_stats = carrier_stats.sort_values("accuracy_rate", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(
            carrier_stats.rename(columns={
                "total_with_edd": "# Parcels",
                "avg_deviation": "Avg Dev (days)",
                "std_deviation": "Std Dev",
                "on_time": "On-Time Count",
                "accuracy_rate": "Accuracy (%)"
            }).style.highlight_max(subset=["Accuracy (%)"], color="lightgreen"),
            use_container_width=True,
        )
        if not carrier_stats.empty:
            best = carrier_stats.iloc[0]
            st.success(f"🏆 Top Carrier: {carrier_stats.index[0]} — {best['accuracy_rate']}% accuracy")
    with c2:
        fig = px.bar(
            carrier_stats.reset_index(),
            x="carrier_name",
            y="accuracy_rate",
            title="Carrier EDD Accuracy (%)",
            labels={"carrier_name": "Carrier", "accuracy_rate": "Accuracy (%)"},
            color="accuracy_rate", color_continuous_scale="RdYlGn",
        )
        fig.update_layout(showlegend=False, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 EDD Deviation Distribution")
    c1, c2 = st.columns(2)
    with c1:
        hist = px.histogram(
            edd_valid, x="edd_accuracy_days", nbins=20,
            title="Distribution of EDD Deviations",
            labels={"edd_accuracy_days": "Days from EDD", "count": "# Parcels"},
        )
        hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="On-Time")
        st.plotly_chart(hist, use_container_width=True)
    with c2:
        box = px.box(
            edd_valid, x="carrier_name", y="edd_accuracy_days",
            title="EDD Deviation by Carrier",
            labels={"carrier_name": "Carrier", "edd_accuracy_days": "Days from EDD"},
        )
        box.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="On-Time")
        st.plotly_chart(box, use_container_width=True)

# ---------------------------------------------------------
# 4) PART 3: EXPLORATION & OPERATIONAL INSIGHTS
# ---------------------------------------------------------
# ---------------------------------------------------------
def part3_data_exploration(parcel_df: pd.DataFrame, log_df: pd.DataFrame) -> None:
    """
    Render Part 3: exploratory data analysis and operational insights.
    """
    st.markdown(
        '<div style="font-size:1.8rem; font-weight:bold; color:#2e8b57; margin:1rem 0; padding:0.5rem;' 
        'background-color:#f0f8f0; border-left:4px solid #2e8b57;">'
        '🔍 Part 3: Data Exploration & Operational Insights</div>',
        unsafe_allow_html=True,
    )

    # Q1: Completed deliveries
    st.subheader("❓ Q1: How many parcels completed their delivery journey?")
    total = len(parcel_df)
    delivered = parcel_df[parcel_df["is_delivered"] == True].shape[0]
    delivery_rate = delivered / total * 100 if total else 0
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("✅ Completed", f"{delivered:,}")
    with c2:
        st.metric("📦 Total", f"{total:,}")
    with c3:
        st.metric("📊 Rate", f"{delivery_rate:.1f}%")

    st.markdown("---")

    # Q2: First attempt success
    st.subheader("❓ Q2: Is parcel always delivered on its first attempt?")
    df_delivered = parcel_df[parcel_df["is_delivered"] == True].copy()
    if df_delivered.empty:
        st.info("No delivered parcels to analyze.")
        first_rate = None
    else:
        first_succ_count = df_delivered[
            df_delivered["first_attempt_date"] == df_delivered["final_delivery_date"]
        ].shape[0]
        first_rate = first_succ_count / len(df_delivered) * 100
        c1, c2 = st.columns(2)
        with c1:
            st.metric("🎯 First-Attempt Success", f"{first_succ_count:,}", f"{first_rate:.1f}% of delivered")
            if first_rate < 50:
                st.error("❌ Low first-attempt success")
            elif first_rate < 80:
                st.warning("⚠️ Moderate first-attempt success")
            else:
                st.success("✅ High first-attempt success")
        with c2:
            carrier_rate = (
                df_delivered.groupby("carrier_name").apply(
                    lambda g: g[g["first_attempt_date"] == g["final_delivery_date"]].shape[0] / g.shape[0] * 100
                )
                .reset_index(name="success_rate")
                .sort_values("success_rate", ascending=False)
            )
            fig = px.bar(
                carrier_rate,
                x="carrier_name",
                y="success_rate",
                title="First-Attempt Success by Carrier",
                labels={"carrier_name": "Carrier", "success_rate": "% Success"},
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Q3: EDD delta analysis (using edd_final for all carriers)
    st.subheader("❓ Q3: Delta in EDD by Carrier")
    df_edd = parcel_df[
        (parcel_df["edd_final"].notna())
        & (parcel_df["final_delivery_date"].notna())
        & (parcel_df["is_delivered"] == True)
    ].copy()
    if df_edd.empty:
        st.info("No parcels with edd_final and final delivery date.")
    else:
        df_edd["edd_accuracy_days"] = (df_edd["final_delivery_date"] - df_edd["edd_final"]).dt.days
        # Exclude extreme deviations beyond ±365 days for summary
        df_edd = df_edd[df_edd["edd_accuracy_days"].abs() <= 365]
        delta_stats = df_edd.groupby("carrier_name")["edd_accuracy_days"].agg(["mean", "median", "std"]).round(2)
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(delta_stats, use_container_width=True)
        with c2:
            fig = px.bar(
                delta_stats.reset_index(),
                x="carrier_name",
                y="mean",
                title="Average EDD Deviation by Carrier",
                labels={"carrier_name": "Carrier", "mean": "Avg Deviation (days)"},
                color="mean", color_continuous_scale="RdYlBu_r",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="On-Time")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Q4: EDD coverage (using edd_final for all carriers)
    st.subheader("❓ Q4: EDD Coverage by Carrier")
    cover_stats = parcel_df.groupby("carrier_name").agg(
        total_parcels=("parcel_id", "count"),
        with_edd=("edd_final", lambda x: x.notna().sum()),
    )
    cover_stats["coverage_rate"] = (cover_stats["with_edd"] / cover_stats["total_parcels"] * 100).round(1)
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(cover_stats, use_container_width=True)
    with c2:
        fig2 = px.bar(
            cover_stats.reset_index(),
            x="carrier_name",
            y="coverage_rate",
            title="EDD Coverage Rate by Carrier",
            labels={"carrier_name": "Carrier", "coverage_rate": "% Coverage"},
        )
        st.plotly_chart(fig2, use_container_width=True)
    overall_cov = parcel_df["edd_final"].notna().mean() * 100
    st.info(f"Overall EDD Coverage: {overall_cov:.1f}%")

    # Q5: Efficacy of Subsequent Deliveries
    st.markdown("---")
    st.subheader("❓ Q5: Efficacy of Subsequent Delivery Attempts")
    # Identify parcels that required more than one delivery attempt
    df_multiple = parcel_df[(parcel_df["first_attempt_date"].notna()) & (parcel_df["final_delivery_date"].notna()) & (parcel_df["first_attempt_date"] != parcel_df["final_delivery_date"])]
    if df_multiple.empty:
        st.info("No parcels required more than one delivery attempt.")
    else:
        # Compute delay between first attempt and final delivery
        df_multiple["attempt_delay_days"] = (df_multiple["final_delivery_date"] - df_multiple["first_attempt_date"]).dt.days
        # Overall metrics
        total_multiple = len(df_multiple)
        avg_delay = df_multiple["attempt_delay_days"].mean().round(1)
        st.metric("📦 Parcels Needing Multiple Attempts", f"{total_multiple:,}")
        st.metric("⏱️ Avg Delay After First Attempt", f"{avg_delay} days")
        # Carrier-level stats
        carrier_delays = df_multiple.groupby("carrier_name")["attempt_delay_days"].agg(["count", "mean"]).rename(columns={"count": "# Parcels", "mean": "Avg Delay (days)"}).round(1)
        st.dataframe(carrier_delays, use_container_width=True)
        # Distribution plot
        fig_delay = px.histogram(
            df_multiple,
            x="attempt_delay_days",
            nbins=20,
            title="Distribution of Delay Between First Attempt and Final Delivery",
            labels={"attempt_delay_days": "Days Delay", "count": "# Parcels"},
        )
        st.plotly_chart(fig_delay, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 Additional Insights & Recommendations")
    # High log activity
    logs_per_parcel = log_df.groupby("parcel_id").size()
    threshold = logs_per_parcel.quantile(0.95)
    high_activity_count = (logs_per_parcel > threshold).sum()
    if high_activity_count > 0:
        st.warning(f"{high_activity_count:,} parcels show excessive log activity (≥95th percentile). Potential data or process issues.")
    if overall_cov < 80:
        st.warning("EDD coverage is below 80%. Consider improving EDD processes.")
    if 'first_rate' in locals() and first_rate is not None and first_rate < 70:
        st.warning("First-attempt success is below 70%. Investigate common failure causes.")

    st.subheader("💼 Recommended Actions")
    recommendations = [
        "🎯 Focus on improving first-attempt success: analyze common failure reasons in logs and address validation.",
        "📊 Enhance EDD data coverage: ensure all carriers consistently log expected_time fields to reduce missing EDDs.",
        "🔍 Investigate parcels with excessive log activity: identify operational bottlenecks or data inconsistencies.",
        "🛠️ Standardize EDD calculation: align on a single approach (log-based midpoint or fallback).",
        "📈 Monitor EDD accuracy over time: set up periodic checks to track improvements and identify regressions.",
    ]
    for rec in recommendations:
        st.info(rec)



# ---------------------------------------------------------
# 5) MAIN APP
# ---------------------------------------------------------

# ---------------------------------------------------------
def main():
    st.markdown(
        """
        <style>
            .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<h1 class="main-header">📦 ParcelPerform EDD Dashboard</h1>', unsafe_allow_html=True)

    # Load data once
    with st.spinner("Loading data..."):
        parcel_df, log_df = load_data()
    if parcel_df is None or log_df is None:
        return

    # Compute log-based EDD fields
    parcel_with_edd = compute_edd_fields(parcel_df, log_df)

    # Render each section sequentially
    render_part1()
    st.markdown("---")
    render_part2(parcel_with_edd)
    st.markdown("---")
    part3_data_exploration(parcel_with_edd, log_df)
    st.markdown("---")
    st.markdown("**📝 Data Analyst Take-Home (All Parts) | Built with Pandas**")

if __name__ == "__main__":
    main()
