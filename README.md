# ParcelPerform EDD Dashboard

An interactive Streamlit application to analyze parcel delivery data, compute Estimated Delivery Dates (EDDs), evaluate EDD accuracy, and surface key operational insights.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Overview](#data-overview)
3. [Part 1: SQL Query Snippets](#part-1-sql-query-snippets)
4. [Part 2: EDD Extraction & Accuracy Methodology](#part-2-edd-extraction--accuracy-methodology)

   * [Log-Based EDD Extraction](#log-based-edd-extraction)
   * [Fallback EDD Calculation](#fallback-edd-calculation)
   * [Accuracy Metrics & Outlier Handling](#accuracy-metrics--outlier-handling)
5. [Part 3: Exploratory & Operational Insights](#part-3-exploratory--operational-insights)

   * [Q1: Delivery Completion Rate](#q1-delivery-completion-rate)
   * [Q2: First-Attempt Success Rate](#q2-first-attempt-success-rate)
   * [Q3: EDD Delta by Carrier](#q3-edd-delta-by-carrier)
   * [Q4: EDD Coverage by Carrier](#q4-edd-coverage-by-carrier)
   * [Q5: Efficacy of Subsequent Delivery Attempts](#q5-efficacy-of-subsequent-delivery-attempts)
6. [Installation & Usage](#installation--usage)
7. [Project Structure](#project-structure)
8. [Dependencies](#dependencies)
9. [Data Assumptions & Considerations](#data-assumptions--considerations)
10. [License](#license)

---

## Introduction

ParcelPerform powers e-commerce merchants with real-time tracking, analytics, and predictive insights for parcel shipments. This repository demonstrates a take-home exercise in which we:

1. Craft SQL queries to answer fundamental parcel‐log questions (Part 1).
2. Extract and compute EDDs from carrier logs, apply a fallback, measure accuracy, and highlight outliers (Part 2).
3. Explore operational metrics—completion rates, first‐attempt success, multi‐attempt efficacy—and surface carrier performance (Part 3).

The final deliverable is a single‐page Streamlit dashboard (`dashboard.py`) that ties all analyses together, enabling an interactive, no‐reload experience.

---

## Data Overview

We work with two Parquet files:

* **`parcel_table.pqt`** (50,083 rows):

  * `parcel_id` (int, primary key)
  * `carrier_name` (string)
  * `picked_up_date` (timestamp)
  * `out_for_delivery_date` (timestamp)
  * `first_attempt_date` (timestamp)
  * `final_delivery_date` (timestamp)
  * `origin_country` (string)
  * `destination_country` (string)
  * `is_delivered` (boolean)

* **`log_table.pqt`** (94,771 rows):

  * `log_id` (int, primary key)
  * `parcel_id` (int, foreign key)
  * `raw_log_description` (string)
  * `log_key` (string)
  * `log_timestamp` (timestamp)
  * `additional_params` (string containing JSON)

The JSON in `additional_params` may include keys like `old_parcel_expected_time_first_start`, `new_parcel_expected_time_latest_end`, etc., representing carrier‐provided EDD ranges. Our goal is to extract the latest matching “\*\_start” / “\*\_end” pair per parcel log and take the midpoint as a log‐based EDD.

All timestamp columns are converted to timezone-naive UTC upon loading.

---

## Part 1: SQL Query Snippets

> **Purpose**: Provide SQL examples (no live computation) to demonstrate how an analyst would retrieve key metrics from a relational database if `parcel_table` and `log_table` were tables in a warehouse.

### 1. Average, Median, and 90th Percentile Transit Time for Domestic Shipments (2024)

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
WHERE origin_country = destination_country   -- domestic only
  AND YEAR(picked_up_date) = 2024
  AND is_delivered = TRUE
  AND final_delivery_date IS NOT NULL
  AND picked_up_date IS NOT NULL
GROUP BY origin_country, destination_country
ORDER BY avg_days;
```

* **Explanation**:

  * Filters for shipments where origin = destination and pickup in 2024.
  * Uses `DATEDIFF` (final – pickup) to compute transit days.
  * `AVG`, `PERCENTILE_CONT(0.5)` (median), and `PERCENTILE_CONT(0.9)` give key percentiles.

---

### 2. Maximum Transit Time & Parcel Count

```sql
WITH transit_times AS (
    SELECT
        parcel_id,
        DATEDIFF(final_delivery_date, picked_up_date) AS transit_days
    FROM parcel_table
    WHERE is_delivered = TRUE
      AND final_delivery_date IS NOT NULL
      AND picked_up_date IS NOT NULL
)
SELECT
    MAX(transit_days)                  AS max_transit_days,
    COUNT(*) FILTER (WHERE transit_days = MAX(transit_days)) AS parcel_count
FROM transit_times;
```

* **Explanation**:

  * First subquery (`transit_times`) computes the transit days for each delivered parcel.
  * The outer query finds `MAX(transit_days)` and counts how many parcels share that maximum.

---

### 3. Top 2 Carriers by Volume per Trade Lane

```sql
WITH carrier_volume AS (
    SELECT
        origin_country,
        destination_country,
        carrier_name,
        COUNT(*)                        AS parcel_volume,
        ROW_NUMBER() OVER (
            PARTITION BY origin_country, destination_country
            ORDER BY COUNT(*) DESC
        )                                AS rn
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

* **Explanation**:

  * Groups by `(origin, destination, carrier_name)` to count volumes.
  * Uses `ROW_NUMBER()` partitioned by trade lane, ordered by volume desc.
  * Filters to the top 2 carriers per lane.

---

### 4. Delivered Parcels with No Log Entries

```sql
SELECT p.*
FROM parcel_table p
LEFT JOIN log_table l
  ON p.parcel_id = l.parcel_id
WHERE p.is_delivered = TRUE
  AND l.parcel_id IS NULL;
```

* **Explanation**:

  * Left join `parcel_table` to `log_table` on `parcel_id`.
  * Filters for `is_delivered = TRUE` but `l.parcel_id IS NULL`, indicating no log rows.

---

### 5. Parcels Handled by Multiple Distinct Carriers

```sql
SELECT
    parcel_id,
    STRING_AGG(DISTINCT carrier_name, '; ') AS list_of_carrier
FROM parcel_table
GROUP BY parcel_id
HAVING COUNT(DISTINCT carrier_name) > 1;
```

* **Explanation**:

  * Groups by `parcel_id`, aggregates unique `carrier_name` values into a semicolon-separated string.
  * `HAVING` ensures only those with more than one distinct carrier appear.

---

## Part 2: EDD Extraction & Accuracy Methodology

In this section, we compute a final Estimated Delivery Date (`edd_final`) for each parcel and measure how closely carriers meet that estimate. We combine:

1. **Log‐Based EDD** (if present).
2. **Fallback EDD** = `picked_up_date` + (carrier’s median transit days).

### Log-Based EDD Extraction

1. **Parse JSON**:

   * For each `parcel_id` in the `log_table`, load the `additional_params` JSON.
   * Identify keys ending in `"_start"` vs `"_end"` that contain `"expected_time_"`.
   * Examples:

     * `old_parcel_expected_time_first_start` & `old_parcel_expected_time_first_end`
     * `new_parcel_expected_time_latest_start` & `new_parcel_expected_time_latest_end`

2. **Pairing**:

   * For each matching `*_start` / `*_end` pair, parse the timestamps to naive `pd.Timestamp`.
   * Keep `(start, end, log_timestamp)` and, after collecting all pairs for a given `parcel_id`, sort by `log_timestamp` ascending.
   * Use the **latest** pair’s midpoint as:

     ```
     edd_mid_log = start_datetime + (end_datetime − start_datetime) / 2
     ```

3. **Result**:

   * Build a small DataFrame (`edd_log_df`) mapping `parcel_id → edd_mid_log`.

### Fallback EDD Calculation

If no log‐based EDD exists or the carrier did not provide any `expected_time_` keys:

1. **Compute Median Transit per Carrier**:

   * From all parcels where `is_delivered = TRUE`, `picked_up_date` & `final_delivery_date` are nonnull:

     ```
     transit_days = (final_delivery_date − picked_up_date).days
     median_transit_days_by_carrier = median(transit_days)  for each carrier_name
     ```
2. **Fallback Logic**:

   * If `edd_mid_log` is present, use it directly as `edd_final`.
   * Otherwise, if `picked_up_date` is nonnull, set:

     ```
     edd_final = picked_up_date + timedelta(days=median_transit_days_by_carrier[carrier_name])
     ```
   * If neither applies (i.e., no log EDD and no valid pickup), leave `edd_final = NaT`.

After merging into the main `parcel_df`, every row has:

* `edd_mid_log` (possibly `NaT`).
* `edd_final` (computed via either log midpoint or fallback).

### Accuracy Metrics & Outlier Handling

1. **Filter for Valid Comparison**:
   Keep only parcels where all three conditions hold:

   * `edd_final` is not null.
   * `final_delivery_date` is not null.
   * `is_delivered = TRUE`.

2. **Compute Deviation**:

   ```
   edd_accuracy_days = (final_delivery_date − edd_final).days
   ```

   A positive value means the parcel arrived **after** the estimated date; negative means **before**.

3. **Show Raw “UNKNOWN” Carrier Deviations**:

   * Before discarding anomalies, display all rows where `carrier_name == "UNKNOWN"` and `edd_accuracy_days` is implausible.
   * This ensures we do not “hide” any data; analysts can inspect raw outliers.

4. **Exclude Extreme Outliers for Metrics**:

   * Any `|edd_accuracy_days| > 365` is considered a data anomaly and excluded from summary metrics (but still shown in the raw outlier section).
   * This threshold captures data‐entry errors (e.g., missing zeroes or Unix‐epoch fallbacks).

5. **Overall Accuracy (±1‐Day Tolerance)**:

   * Let `N_total` = number of parcels passing the above filters (`|edd_accuracy_days| <= 365`).
   * Let `N_on_time` = count where `|edd_accuracy_days| <= 1`.
   * **Overall Accuracy Rate** = `N_on_time / N_total × 100%`.

6. **Carrier‐Level Performance**:
   For each `carrier_name`, compute:

   * `total_with_edd` = count of parcels contributing to accuracy (post‐filter).
   * `avg_deviation` = mean of `edd_accuracy_days`.
   * `std_deviation` = standard deviation.
   * `on_time_count` = count where `|edd_accuracy_days| <= 1`.
   * **Accuracy (%)** = `on_time_count / total_with_edd × 100`.

7. **Visualization**:

   * Bar chart of carrier accuracy rates (colored by % accurate).
   * Histogram of `edd_accuracy_days` across all carriers, with a red dashed line at 0 (on‐time).
   * Box plots showing deviation distribution by carrier, with y=0 as on‐time line.

---

## Part 3: Exploratory & Operational Insights

Using the same `parcel_with_edd` (containing `edd_final`), we explore five core questions:

### Q1: Delivery Completion Rate

* **Metric**:

  * `total_parcels = len(parcel_df)`
  * `delivered_count = count where is_delivered = TRUE`
  * **Completion Rate** = `delivered_count / total_parcels × 100%`.

* **Rationale**:
  Provides a high‐level view of how many parcels in the dataset actually reached their destination.

---

### Q2: First‐Attempt Success Rate

* **Metric** (over delivered parcels):

  * Let `df_delivered = parcel_df[is_delivered == TRUE]`.
  * `first_success_count = count where first_attempt_date == final_delivery_date`.
  * **First‐Attempt Success Rate** = `first_success_count / len(df_delivered) × 100%`.

* **Per‐Carrier Breakdown**:

  * Group `df_delivered` by `carrier_name`.
  * For each group, compute the same ratio.
  * Visualize with a bar chart of `% success`.

* **Rationale**:
  First‐attempt failures incur extra costs (redelivery). Highlighting carriers with low first‐attempt rates guides process improvements (e.g., address validation, driver training).

---

### Q3: EDD Delta by Carrier

* **Using `edd_final`** (Part 2 logic) for all carriers:

  * Filter delivered parcels where both `edd_final` and `final_delivery_date` are nonnull.
  * Compute `edd_accuracy_days = (final_delivery_date − edd_final).days`.
  * Exclude `|edd_accuracy_days| > 365` as outliers for summary.
  * **Carrier‐Level Stats**:

    * `mean_deviation`, `median_deviation`, `std_deviation`.

* **Visual**:

  * Bar chart of each carrier’s average deviation.
  * Horizontal line at 0 days (on‐time benchmark).

* **Rationale**:
  Comparing average early/late delivery tendencies per carrier helps identify which carriers are systematically over‐ or under‐promising.

---

### Q4: EDD Coverage by Carrier

* **Definition**:

  * For each carrier, count:

    * `total_parcels`: total shipments for that carrier.
    * `with_edd`: count where `edd_final` is nonnull.
  * **Coverage Rate** = `with_edd / total_parcels × 100%`.

* **Visual**:

  * Table of `total_parcels`, `with_edd`, `coverage_rate` by carrier.
  * Bar chart of coverage rate per carrier.
  * Overall EDD coverage across all carriers (info box).

* **Rationale**:
  Low EDD coverage means too many parcels lack an estimate, reducing trust in the system.

---

### Q5: Efficacy of Subsequent Delivery Attempts

* **Goal**: Understand how much additional delay occurs when a parcel is not delivered on the first attempt.

1. **Identify Multi‐Attempt Parcels**:

   * Filter `parcel_df` where:

     * `is_delivered = TRUE`
     * `first_attempt_date` and `final_delivery_date` are both nonnull
     * `first_attempt_date != final_delivery_date`

2. **Compute Delay After First Attempt**:

   ```
   attempt_delay_days = (final_delivery_date − first_attempt_date).days
   ```

3. **Overall Metrics**:

   * `total_multiple = count of multi‐attempt parcels`
   * `avg_delay = mean(attempt_delay_days)`

4. **Carrier‐Level Stats**:

   * Group by `carrier_name`, compute:

     * `# Parcels` (multi‐attempt)
     * `Avg Delay (days)`

5. **Visual**:

   * DataFrame showing per‐carrier counts & average delay.
   * Histogram of `attempt_delay_days` to see distribution.

* **Rationale**:
  High multi‐attempt rates or large delays after first attempt indicate operational inefficiencies—missed delivery windows, address errors, or scheduling challenges.

---

## Installation & Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/0xanp/pp_takehome.git
   cd parcelperform-edd-dashboard
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Unix/macOS
   # On Windows:
   # venv\\Scripts\\activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Place data files in project root**

   * `parcel_table.pqt`
   * `log_table.pqt`

5. **Run the Streamlit app**

   ```bash
   streamlit run dashboard.py
   ```

6. **Interact**:

   * Scroll through Part 1 to review SQL snippets.
   * Explore Part 2’s EDD accuracy metrics and outlier listings.
   * Dive into Part 3’s operational insights (delivery rates, first‐attempt success, multi‐attempt efficacy).

---

## Project Structure

```
parcelperform-edd-dashboard/
├── dashboard.py         # Main Streamlit application
├── parcel_table.pqt     # Parquet file: primary parcel data
├── log_table.pqt        # Parquet file: log entries with JSON in additional_params
├── requirements.txt     # Python dependencies (pandas, numpy, plotly, streamlit, etc.)
└── README.md            # Detailed explanation (this file)
```

---

## Dependencies

* Python ≥ 3.8
* pandas
* numpy
* plotly
* streamlit

To install:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```
pandas>=1.3
numpy>=1.20
plotly>=5.0
streamlit>=1.0
```

---

## Data Assumptions & Considerations

1. **Timezone Handling**

   * All timestamp columns (`picked_up_date`, `log_timestamp`, etc.) are converted to UTC naive.
   * This ensures arithmetic (like `final_delivery_date − edd_final`) is straightforward.

2. **EDD Extraction**

   * We rely on carriers populating one of the JSON keys:

     * `*_expected_time_first_start` / `*_expected_time_first_end`
     * `*_expected_time_latest_start` / `*_expected_time_latest_end`
   * If multiple EDD keys appear, we choose the latest by `log_timestamp`.

3. **Fallback Calculation**

   * If no log‐based EDD exists, we compute the carrier’s median transit (in days) using only successfully delivered parcels with valid pickup/delivery dates.
   * Set `edd_final = picked_up_date + median_transit_days`.
   * If `picked_up_date` is null, we cannot compute a fallback; `edd_final` remains `NaT`.

4. **Outlier Filtering**

   * Any `|edd_accuracy_days| > 365` is considered a data anomaly and excluded from summary metrics (but still shown in raw outlier tables).
   * This threshold is high enough to capture data‐entry errors (e.g., missing zeroes or Unix‐epoch fallbacks).

5. **“UNKNOWN” Carrier Handling**

   * No special‐case; treat “UNKNOWN” like any other carrier.
   * If no log EDD exists and pickup is valid, we compute `edd_final = pickup + median_transit`.
   * The raw outlier table will show if “UNKNOWN” yields extreme deviations, indicating missing or incorrect data.

6. **First‐Attempt vs. Final Delivery**

   * We assume `first_attempt_date == final_delivery_date` means success on the first attempt.
   * If they differ, the parcel required a subsequent attempt.

---

## License

This project is licensed under the MIT License.
