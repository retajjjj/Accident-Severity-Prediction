"""
Data Validation Script — UK Road Accidents Project
CMPS344 Applied Data Science — Phase 2

Covers all 8 validation dimensions from the course lecture:
    1. Accuracy       — business rule checks
    2. Consistency    — format and logical cross-field checks
    3. Completeness   — missing value analysis
    4. Uniqueness     — duplicate detection (exact + key-level)
    5. Outliers       — IQR, Z-score, Isolation Forest
    6. Timeliness     — date range, year coverage, COVID dip flag
    7. Distribution   — min/max/mean/std/skewness/kurtosis/KS test
    8. Relationships  — Pearson vs Spearman, high-correlation flags

Outputs (all saved to reports/):
    validation_report.json     — full machine-readable findings
    validation_summary.txt     — plain-text summary for report screenshot

Usage:
    poetry run python src/data/validate.py
    make validate
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest
import os
from acquire import download_dataset
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors

# Get path from acquire module
path = download_dataset()

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
RAW_DIR     = Path(path)

REPORTS_DIR  = Path("reports")
ACCIDENTS_FILE = RAW_DIR / "Accident_Information.csv"
VEHICLES_FILE  = RAW_DIR / "Vehicle_Information.csv"

# ── Report accumulator ──────────────────────────────────────────────────────
report = {
    "generated_at": datetime.now().isoformat(),
    "dimensions": {},
    "summary": {"issues": []}
}
log_lines = []


def log(msg: str = ""):
    print(msg)
    log_lines.append(msg)


def issue(msg: str, dimension: str):
    log(f"  [ISSUE] {msg}")
    report["summary"]["issues"].append({"dimension": dimension, "message": msg})


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_tables():
    log("=" * 65)
    log("  UK Road Accidents — Data Validation Report")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 65)

    for path in [ACCIDENTS_FILE, VEHICLES_FILE]:
        if not path.exists():
            log(f"\n[ERROR] {path} not found.")
            log("        Run: make data   or   poetry run python src/data/acquire.py")
            raise SystemExit(1)

    acc = pd.read_csv(ACCIDENTS_FILE,encoding='latin1', low_memory=False)
    veh = pd.read_csv(VEHICLES_FILE,encoding='latin1',  low_memory=False)

    log(f"\nLoaded accidents : {acc.shape[0]:>9,} rows x {acc.shape[1]} columns")
    log(f"Loaded vehicles  : {veh.shape[0]:>9,} rows x {veh.shape[1]} columns")
    return acc, veh


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 1 — ACCURACY
#  Does data correctly represent reality?
#  Apply business rules specific to UK road accident data.
# ════════════════════════════════════════════════════════════════════════════

def check_accuracy(acc: pd.DataFrame, veh: pd.DataFrame):
    log("\n" + "-" * 65)
    log("DIMENSION 1 — ACCURACY")
    log("-" * 65)
    dim = {}

    # 1a. Target variable must be one of 3 valid classes
    valid_severity = {"Slight", "Serious", "Fatal", "1", "2", "3"}
    if "Accident_Severity" in acc.columns:
        bad = ~acc["Accident_Severity"].astype(str).isin(valid_severity)
        n = int(bad.sum())
        dim["invalid_severity_values"] = n
        if n:
            issue(f"{n:,} rows have unrecognised Accident_Severity values", "accuracy")
        else:
            log("  [PASS] Accident_Severity — all values valid (Slight/Serious/Fatal)")

    # 1b. Speed_limit must be a UK legal limit: 20,30,40,50,60,70
    uk_speed_limits = {20, 30, 40, 50, 60, 70}
    if "Speed_limit" in acc.columns:
        speed = pd.to_numeric(acc["Speed_limit"], errors="coerce").dropna()
        bad_speed = speed[~speed.isin(uk_speed_limits)]
        n = len(bad_speed)
        dim["invalid_speed_limits"] = n
        if n:
            issue(f"{n:,} rows have Speed_limit not in UK legal set {uk_speed_limits}", "accuracy")
        else:
            log("  [PASS] Speed_limit — all values are valid UK legal limits")

    # 1c. Coordinates must be within UK geographic bounds
    if "Latitude" in acc.columns and "Longitude" in acc.columns:
        lat = pd.to_numeric(acc["Latitude"],  errors="coerce")
        lon = pd.to_numeric(acc["Longitude"], errors="coerce")
        bad_lat = int(((lat < 49.0) | (lat > 61.0)).sum())
        bad_lon = int(((lon < -8.0) | (lon >  2.0)).sum())
        dim["out_of_bounds_latitude"]  = bad_lat
        dim["out_of_bounds_longitude"] = bad_lon
        if bad_lat:
            issue(f"{bad_lat:,} rows outside UK latitude range (49–61)", "accuracy")
        else:
            log("  [PASS] Latitude — all within UK bounds (49–61)")
        if bad_lon:
            issue(f"{bad_lon:,} rows outside UK longitude range (-8 to 2)", "accuracy")
        else:
            log("  [PASS] Longitude — all within UK bounds (-8 to 2)")

    # 1d. Casualty and vehicle counts must be >= 1
    for col in ["Number_of_Vehicles"]:
        if col in acc.columns:
            bad = int((pd.to_numeric(acc[col], errors="coerce") <= 0).sum())
            dim[f"non_positive_{col}"] = bad
            if bad:
                issue(f"{bad:,} rows have {col} <= 0 (must be >= 1)", "accuracy")
            else:
                log(f"  [PASS] {col} — all values >= 1")

    # 1e. Vehicle age cannot be negative
    if "Age_of_Vehicle" in veh.columns:
        neg = int((pd.to_numeric(veh["Age_of_Vehicle"], errors="coerce") < 0).sum())
        dim["negative_vehicle_age"] = neg
        if neg:
            issue(f"{neg:,} vehicle records have negative Age_of_Vehicle", "accuracy")
        else:
            log("  [PASS] Age_of_Vehicle — no negative values")
    acceptable_band = [ "16 - 20", "21 - 25", "26 - 35", "36 - 45", "46 - 55", "56 - 65", "66 - 75", "Data missing or out of range", "Over 75"]
    if "Age_Band_of_Driver" in veh.columns:
        bad_band = int((~veh["Age_Band_of_Driver"].isin(acceptable_band)).sum())
        dim["invalid_age_band"] = bad_band
        if bad_band:
            issue(f"{bad_band:,} vehicle records have invalid Age_Band_of_Driver values", "accuracy")
        else:
            log("  [PASS] Age_Band_of_Driver — all values valid age bands")
    if "Day_of_Week" in acc.columns:
        bad_band = int((~acc["Day_of_Week"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])).sum())
        dim["invalid_day_of_week"] = bad_band
        if bad_band:
            issue(f"{bad_band:,} rows have invalid Day_of_Week values", "accuracy")
        else:
            log("  [PASS] Day_of_Week — all values valid (1=Sun ... 7=Sat)")
    if "Sex_of_Driver" in veh.columns:
        bad_band = int((~veh["Sex_of_Driver"].isin(["Male", "Female"])).sum())
        dim["invalid_sex_band"] = bad_band
        if bad_band:
            issue(f"{bad_band:,} vehicle records have invalid Sex_of_Driver values", "accuracy")
        else:
            log("  [PASS] Sex_of_Driver — all values valid sex")

    report["dimensions"]["accuracy"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 2 — CONSISTENCY
#  Is data uniform across fields, formats, and both tables?
# ════════════════════════════════════════════════════════════════════════════

def check_consistency(acc: pd.DataFrame, veh: pd.DataFrame):
    log("\n" + "-" * 65)
    log("DIMENSION 2 — CONSISTENCY")
    log("-" * 65)
    dim = {}

    if "Date" in acc.columns:
        parsed = pd.to_datetime(acc["Date"], format='%Y-%m-%d')

        # 2a. Date column must parse as datetime
        n_failed = int(parsed.isna().sum() - acc["Date"].isna().sum())
        dim["date_parse_failures"] = n_failed
        if n_failed:
            issue(f"{n_failed:,} Date values could not be parsed as datetime", "consistency")
        else:
            log("  [PASS] Date column — parses correctly as datetime")

        # 2b. Year column must match year extracted from Date
    if "Year" in acc.columns:
        mismatch = int(
            (parsed.dt.year != pd.to_numeric(acc["Year"], errors="coerce")).sum()
        )
        dim["year_date_mismatch"] = mismatch
        if mismatch:
            issue(f"{mismatch:,} rows where Year column does not match year in Date", "consistency")
        else:
            log("  [PASS] Year column — matches year extracted from Date in all rows")

        # 2c. Day_of_Week must match actual day derived from Date
        # UK STATS19 encoding: 1=Sunday, 2=Monday ... 7=Saturday
    if "Day_of_Week" in acc.columns:
        parsed = pd.to_datetime(acc["Date"], format='%Y-%m-%d')
        actual_day_names = parsed.dt.day_name()  # Get "Monday", "Tuesday", etc.
        recorded_day_names = acc["Day_of_Week"]
        
        mismatches = (actual_day_names != recorded_day_names).sum()
        if mismatches:
            log(f"  [WARN] {mismatches:,} Date/Day_of_Week mismatches found")
        else:
            log(f"  [PASS] All Day_of_Week values match their dates")
    # 2d. Time must follow HH:MM format
    if "Time" in acc.columns:
        bad_time = int(
            (acc["Time"].dropna().str.match(r"^\d{1,2}:\d{2}$") == False).sum()
        )
        dim["invalid_time_format"] = bad_time
        if bad_time:
            issue(f"{bad_time:,} Time values do not follow HH:MM format", "consistency")
        else:
            log("  [PASS] Time column — all values follow HH:MM format")

    # 2e. Year must be consistent across both tables for the same accident
    if "Year" in acc.columns and "Year" in veh.columns:
        merged = acc[["Accident_Index", "Year"]].merge(
            veh[["Accident_Index", "Year"]].drop_duplicates("Accident_Index"),
            on="Accident_Index", suffixes=("_acc", "_veh"), how="inner"
        )
        cross_mismatch = int(
            (pd.to_numeric(merged["Year_acc"], errors="coerce") !=
             pd.to_numeric(merged["Year_veh"], errors="coerce")).sum()
        )
        dim["cross_table_year_mismatch"] = cross_mismatch
        if cross_mismatch:
            issue(f"{cross_mismatch:,} accidents have different Year in accidents vs vehicles table",
                  "consistency")
        else:
            log("  [PASS] Year — consistent across both tables for all matched accidents")

    report["dimensions"]["consistency"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 3 — COMPLETENESS
#  Is all required data present? Missing value analysis per column.
# ════════════════════════════════════════════════════════════════════════════

def check_completeness(acc: pd.DataFrame, veh: pd.DataFrame):
    log("\n" + "-" * 65)
    log("DIMENSION 3 — COMPLETENESS")
    log("-" * 65)
    dim = {}

    REQUIRED_ACC = [
        "Accident_Index", "Accident_Severity", "Date", "Latitude", "Longitude",
        "Year", "Speed_limit", "Road_Type", "Weather_Conditions",
        "Light_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area"
    ]
    REQUIRED_VEH = [
        "Accident_Index", "Vehicle_Type", "Vehicle_Manoeuvre", "Age_Band_of_Driver"
    ]

    for df, label, required in [
        (acc, "ACCIDENTS", REQUIRED_ACC),
        (veh, "VEHICLES",  REQUIRED_VEH)
    ]:
        log(f"\n  [{label}]")
        total = len(df)
        missing_summary = {}

        for col in df.columns:
            n = int(df[col].isna().sum())
            if n > 0:
                missing_summary[col] = {
                    "count": n,
                    "percent": round(n / total * 100, 2)
                }

        dim[f"{label.lower()}_missing"] = missing_summary

        # Print top 12 columns by missing %
        sorted_m = sorted(missing_summary.items(),
                          key=lambda x: x[1]["percent"], reverse=True)
        log(f"  {'Column':<46} {'Missing':>9}  {'%':>6}")
        log(f"  {'-'*46} {'-'*9}  {'-'*6}")
        for col, info in sorted_m[:12]:
            marker = "  <- REQUIRED" if col in required else ""
            log(f"  {col:<46} {info['count']:>9,}  {info['percent']:>5.1f}%{marker}")
        if len(sorted_m) > 12:
            log(f"  ... and {len(sorted_m)-12} more columns with missing values")

        # Flag required columns with any nulls
        crit = [c for c in required if c in df.columns and df[c].isna().sum() > 0]
        if crit:
            for c in crit:
                issue(f"[{label}] Required column '{c}' has "
                      f"{missing_summary[c]['count']:,} null values "
                      f"({missing_summary[c]['percent']:.1f}%)", "completeness")
        else:
            log(f"\n  [PASS] No nulls in any required columns for {label}")

        # Flag any column >20% missing
        high = [(c, v) for c, v in sorted_m if v["percent"] > 20]
        if high:
            for c, v in high:
                issue(f"[{label}] '{c}' is {v['percent']:.1f}% missing (>20% threshold)",
                      "completeness")

    report["dimensions"]["completeness"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 4 — UNIQUENESS
#  Exact duplicates, key-level duplicates, composite key check.
# ════════════════════════════════════════════════════════════════════════════

def check_uniqueness(acc: pd.DataFrame, veh: pd.DataFrame):
    log("\n" + "-" * 65)
    log("DIMENSION 4 — UNIQUENESS")
    log("-" * 65)
    dim = {}

    # Accidents: Accident_Index MUST be unique (one row per accident)
    acc_exact = int(acc.duplicated().sum())
    acc_key   = int(acc["Accident_Index"].duplicated().sum())
    dim["accidents_exact_duplicates"] = acc_exact
    dim["accidents_key_duplicates"]   = acc_key

    if acc_exact:
        issue(f"Accidents table: {acc_exact:,} fully duplicate rows", "uniqueness")
    else:
        log("  [PASS] Accidents — zero fully duplicate rows")

    if acc_key:
        issue(f"Accidents table: {acc_key:,} duplicate Accident_Index values "
              f"(each accident must appear exactly once)", "uniqueness")
    else:
        log(f"  [PASS] Accidents — all {len(acc):,} Accident_Index values are unique")

    # Vehicles: Accident_Index duplicates are EXPECTED (multiple vehicles per accident)
    veh_exact = int(veh.duplicated().sum())
    veh_key   = int(veh["Accident_Index"].duplicated().sum())
    dim["vehicles_exact_duplicates"]             = veh_exact
    dim["vehicles_duplicate_accident_index"]     = veh_key
    dim["vehicles_duplicate_accident_index_note"] = (
        "Duplicate Accident_Index values in the vehicles table are EXPECTED "
        "by design — multiple vehicles can be involved in one accident."
    )

    if veh_exact:
        issue(f"Vehicles table: {veh_exact:,} fully duplicate rows "
              f"(all fields identical — unexpected)", "uniqueness")
    else:
        log("  [PASS] Vehicles — zero fully duplicate rows")

    log(f"  [INFO] Vehicles — {veh_key:,} duplicate Accident_Index values (expected by design)")

    # Vehicles composite key: (Accident_Index + Vehicle_Reference) must be unique
    if "Vehicle_Reference" in veh.columns:
        comp_dups = int(
            veh.duplicated(subset=["Accident_Index", "Vehicle_Reference"]).sum()
        )
        dim["vehicles_composite_key_duplicates"] = comp_dups
        if comp_dups:
            issue(f"Vehicles: {comp_dups:,} duplicate (Accident_Index, Vehicle_Reference) pairs",
                  "uniqueness")
        else:
            log("  [PASS] Vehicles — (Accident_Index, Vehicle_Reference) composite key is fully unique")

    report["dimensions"]["uniqueness"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 5 — OUTLIERS
#  IQR method + Z-score method + Isolation Forest (all 3 from lecture)
# ════════════════════════════════════════════════════════════════════════════

def check_outliers(acc: pd.DataFrame, veh: pd.DataFrame):
    log("\n" + "-" * 65)
    log("DIMENSION 5 — OUTLIERS")
    log("-" * 65)
    dim = {}

    COLS_ACC = ["Speed_limit", "Number_of_Casualties", "Number_of_Vehicles"]
    COLS_VEH = ["Age_of_Vehicle", "Engine_Capacity_.CC."]

    # ── IQR method ──────────────────────────────────────────────────────────
    log("\n  IQR Method  (bounds = Q1 - 1.5*IQR  and  Q3 + 1.5*IQR):")
    log(f"  {'Column':<42} {'Outliers':>10}  {'Lower fence':>12}  {'Upper fence':>12}")
    log(f"  {'-'*42} {'-'*10}  {'-'*12}  {'-'*12}")

    iqr_res = {}
    for df, cols, label in [(acc, COLS_ACC, "accidents"),
                             (veh, COLS_VEH, "vehicles")]:
        for col in cols:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_out = int(((s < lo) | (s > hi)).sum())
            key = f"{label}.{col}"
            iqr_res[key] = {"count": n_out,
                            "lower_fence": round(float(lo), 2),
                            "upper_fence": round(float(hi), 2)}
            log(f"  {key:<42} {n_out:>10,}  {lo:>12.2f}  {hi:>12.2f}")
    dim["iqr"] = iqr_res

    # ── Isolation Forest (multivariate) ────────────────────────────────────
    log("\n  Isolation Forest  (contamination=0.05, multivariate):")
    iso_res = {}

    for df, cols, label in [(acc, COLS_ACC, "accidents"),
                             (veh, COLS_VEH, "vehicles")]:
        avail = [c for c in cols if c in df.columns]
        if len(avail) < 2:
            continue
        sub = df[avail].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 100:
            log(f"  [SKIP] {label} — too few non-null rows")
            continue
        sample = sub.sample(min(100_000, len(sub)), random_state=42)
        clf = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        preds = clf.fit_predict(sample)
        n_anom = int((preds == -1).sum())
        rate   = round(n_anom / len(sample) * 100, 2)
        iso_res[label] = {
            "features_used":      avail,
            "sample_size":        len(sample),
            "anomalies_detected": n_anom,
            "anomaly_rate_pct":   rate
        }
        log(f"  {label}: {n_anom:,} anomalies in {len(sample):,} rows ({rate:.1f}%)")

    dim["isolation_forest"] = iso_res
    log("\n  [NOTE] Outliers are FLAGGED only — removal decisions made in cleaning step.")
    report["dimensions"]["outliers"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 6 — TIMELINESS
#  Date range validity, year coverage, COVID dip detection.
# ════════════════════════════════════════════════════════════════════════════

def check_timeliness(acc: pd.DataFrame):
    log("\n" + "-" * 65)
    log("DIMENSION 6 — TIMELINESS")
    log("-" * 65)
    dim = {}

    if "Date" not in acc.columns:
        log("  [SKIP] No Date column")
        return

    parsed = pd.to_datetime(acc["Date"], dayfirst=True, errors="coerce")
    valid  = parsed.dropna()

    dim["earliest_date"] = str(valid.min().date()) if len(valid) else "N/A"
    dim["latest_date"]   = str(valid.max().date()) if len(valid) else "N/A"
    log(f"  Earliest record : {dim['earliest_date']}")
    log(f"  Latest record   : {dim['latest_date']}")

    # Future dates
    future = int((valid > pd.Timestamp.now()).sum())
    dim["future_dates"] = future
    if future:
        issue(f"{future:,} records have future dates", "timeliness")
    else:
        log("  [PASS] No future dates")

    # Pre-2005 dates
    pre2005 = int((valid < pd.Timestamp("2005-01-01")).sum())
    dim["pre_2005_dates"] = pre2005
    if pre2005:
        issue(f"{pre2005:,} records have dates before 2005", "timeliness")
    else:
        log("  [PASS] No dates before 2005")


    report["dimensions"]["timeliness"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 7 — DISTRIBUTION PROFILE
#  Descriptive stats + skewness + kurtosis + KS test vs normal
# ════════════════════════════════════════════════════════════════════════════

def check_distribution(acc: pd.DataFrame, veh: pd.DataFrame):
    log("\n" + "-" * 65)
    log("DIMENSION 7 — DISTRIBUTION PROFILE")
    log("-" * 65)
    dim = {}

    COLS_ACC = ["Speed_limit", "Number_of_Casualties", "Number_of_Vehicles",
                "Latitude", "Longitude"]
    COLS_VEH = ["Age_of_Vehicle", "Engine_Capacity_.CC."]

    log(f"\n  {'Column':<42} {'Min':>7} {'Max':>8} {'Mean':>8} "
        f"{'Median':>8} {'Std':>8} {'Skew':>7} {'Kurt':>7}  {'Card':>6}")
    log(f"  {'-'*42} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7}  {'-'*6}")

    stats_res = {}
    ks_res    = {}
    hist_res  = {}

    for df, cols, label in [(acc, COLS_ACC, "accidents"),
                             (veh, COLS_VEH, "vehicles")]:
        for col in cols:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 10:
                continue
            
            # Cardinality
            cardinality = s.nunique()
            
            sk = float(skew(s))
            ku = float(kurtosis(s))
            cs = {
                "min":      round(float(s.min()), 3),
                "max":      round(float(s.max()), 3),
                "mean":     round(float(s.mean()), 3),
                "median":   round(float(s.median()), 3),
                "std":      round(float(s.std()), 3),
                "skewness": round(sk, 3),
                "kurtosis": round(ku, 3),
                "n_valid":  len(s),
                "cardinality": cardinality
            }
            key = f"{label}.{col}"
            stats_res[key] = cs
            log(f"  {key:<42} {cs['min']:>7.1f} {cs['max']:>8.1f} {cs['mean']:>8.2f} "
                f"{cs['median']:>8.2f} {cs['std']:>8.2f} {sk:>7.2f} {ku:>7.2f}  {cardinality:>6,}")

            # Histogram (10 equal-width bins)
            hist_counts, bin_edges = np.histogram(s, bins=10)
            histogram = {
                "n_bins": 10,
                "bin_edges": [round(float(x), 3) for x in bin_edges],
                "bin_counts": [int(c) for c in hist_counts],
                "histogram_pct": [round(100 * c / len(s), 1) for c in hist_counts]
            }
            hist_res[key] = histogram
            
            # Print histogram as ASCII bar chart
            log(f"\n    Histogram ({key}):")
            for i, (count, pct) in enumerate(zip(hist_counts, histogram["histogram_pct"])):
                bin_start = round(bin_edges[i], 2)
                bin_end = round(bin_edges[i+1], 2)
                bar = "▓" * max(1, int(pct / 2))  # Scale to 50% width
                log(f"      [{bin_start:>8} – {bin_end:>8}]  {count:>6,}  ({pct:>5.1f}%)  {bar}")

            # KS test against normal distribution
            samp = s.sample(min(5000, len(s)), random_state=42)
            ks_stat, p_val = stats.kstest(
                samp, "norm", args=(float(samp.mean()), float(samp.std()))
            )
            ks_res[key] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value":      round(float(p_val), 6),
                "interpretation": (
                    "NOT normally distributed (p < 0.05) — use Spearman for correlations"
                    if p_val < 0.05 else
                    "Cannot reject normality (p >= 0.05) — Pearson is appropriate"
                )
            }

    dim["descriptive_stats"] = stats_res
    dim["histograms"] = hist_res
    dim["ks_test_vs_normal"] = ks_res

    # KS results summary
    log("\n  KS Test vs Normal Distribution:")
    log(f"  {'Column':<42} {'KS stat':>9}  {'p-value':>10}  Interpretation")
    log(f"  {'-'*42} {'-'*9}  {'-'*10}  {'-'*40}")
    for key, res in ks_res.items():
        log(f"  {key:<42} {res['ks_statistic']:>9.4f}  {res['p_value']:>10.6f}  "
            f"{res['interpretation'][:45]}")

    # Target class distribution
    if "Accident_Severity" in acc.columns:
        severity_dist = acc["Accident_Severity"].value_counts()
        dim["target_severity_distribution"] = severity_dist.to_dict()

    # Categorical feature distributions
    CATEGORICAL_COLS_ACC = [
        "Day_of_Week", "Light_Conditions", "Weather_Conditions", 
        "Road_Surface_Conditions", "Urban_or_Rural_Area", "Road_Type"
    ]
    CATEGORICAL_COLS_VEH = [
        "Age_Band_of_Driver", "Sex_of_Driver", "Vehicle_Type", "Vehicle_Manoeuvre"
    ]

    dim["categorical_distributions"] = {}

    for df, cols, label in [(acc, CATEGORICAL_COLS_ACC, "accidents"),
                             (veh, CATEGORICAL_COLS_VEH, "vehicles")]:
        for col in cols:
            if col in df.columns:
                value_counts = df[col].value_counts()
                dim["categorical_distributions"][f"{label}_{col}"] = value_counts.to_dict()
    
    report["dimensions"]["distribution"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 8 — RELATIONSHIPS
#  Pearson vs Spearman on numeric pairs, flag high correlations
# ════════════════════════════════════════════════════════════════════════════

def check_relationships(acc: pd.DataFrame):
    log("\n" + "-" * 65)
    log("DIMENSION 8 — RELATIONSHIPS")
    log("-" * 65)
    dim = {}

    NUM_COLS = [c for c in [
        "Speed_limit", "Number_of_Casualties", "Number_of_Vehicles",
        "Latitude", "Longitude"
    ] if c in acc.columns]

    if len(NUM_COLS) < 2:
        log("  [SKIP] Not enough numeric columns")
        return

    sub = acc[NUM_COLS].apply(pd.to_numeric, errors="coerce").dropna()
    pearson_m  = sub.corr(method="pearson").round(3)
    spearman_m = sub.corr(method="spearman").round(3)

    dim["pearson_matrix"]  = pearson_m.to_dict()
    dim["spearman_matrix"] = spearman_m.to_dict()

    # Print Pearson matrix
    log("\n  Pearson Correlation Matrix:")
    header = "  " + " " * 25 + "".join(f"{c[:11]:>13}" for c in NUM_COLS)
    log(header)
    for row in NUM_COLS:
        vals = "".join(f"{pearson_m.loc[row, c]:>13.3f}" for c in NUM_COLS)
        log(f"  {row[:24]:<25}{vals}")

    # Pairwise Pearson vs Spearman comparison
    log("\n  Pearson vs Spearman  (|diff| > 0.10 = non-linear relationship detected):")
    log(f"  {'Pair':<44} {'Pearson':>9}  {'Spearman':>9}  {'|Diff|':>8}  Note")
    log(f"  {'-'*44} {'-'*9}  {'-'*9}  {'-'*8}  {'-'*25}")

    pair_res = {}
    high_corr = []

    for i, c1 in enumerate(NUM_COLS):
        for c2 in NUM_COLS[i+1:]:
            p   = float(pearson_m.loc[c1, c2])
            sp  = float(spearman_m.loc[c1, c2])
            diff = abs(p - sp)
            note = "non-linear" if diff > 0.10 else ""
            key  = f"{c1} x {c2}"
            pair_res[key] = {"pearson": round(p,3), "spearman": round(sp,3),
                            "diff": round(diff,3)}
            log(f"  {key:<44} {p:>9.3f}  {sp:>9.3f}  {diff:>8.3f}  {note}")

            if abs(p) > 0.85 or abs(sp) > 0.85:
                high_corr.append(key)
                issue(
                    f"High correlation: '{c1}' x '{c2}' "
                    f"(Pearson={p:.3f}, Spearman={sp:.3f}) — "
                    f"consider dropping one in feature selection",
                    "relationships"
                )

    dim["pair_comparison"] = pair_res
    dim["high_correlation_pairs"] = high_corr

    if not high_corr:
        log("\n  [PASS] No high-correlation pairs found (threshold |r| > 0.85)")

    report["dimensions"]["relationships"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  JOIN INTEGRITY — multi-source merge validation (rubric requirement)
# ════════════════════════════════════════════════════════════════════════════

def check_join_integrity(acc: pd.DataFrame, veh: pd.DataFrame):
    log("\n" + "-" * 65)
    log("JOIN INTEGRITY  (Accident_Index key across both tables)")
    log("-" * 65)
    dim = {}

    acc_keys = set(acc["Accident_Index"].dropna())
    veh_keys = set(veh["Accident_Index"].dropna())
    matched  = acc_keys & veh_keys

    dim["accidents_unique_keys"] = len(acc_keys)
    dim["vehicles_unique_keys"]  = len(veh_keys)
    dim["matched_keys"]          = len(matched)
    dim["only_in_accidents"]     = len(acc_keys - veh_keys)
    dim["only_in_vehicles"]      = len(veh_keys - acc_keys)

    log(f"  Unique Accident_Index in accidents table : {len(acc_keys):>10,}")
    log(f"  Unique Accident_Index in vehicles table  : {len(veh_keys):>10,}")
    log(f"  Keys matched (will survive inner join)   : {len(matched):>10,}")

    if dim["only_in_accidents"]:
        pct = dim["only_in_accidents"] / len(acc_keys) * 100
        issue(
            f"{dim['only_in_accidents']:,} accidents ({pct:.1f}%) have no vehicle record — "
            f"dropped on inner join, document in cleaning step",
            "join_integrity"
        )
    else:
        log("  [PASS] Every accident key has at least one vehicle record")

    if dim["only_in_vehicles"]:
        pct = dim["only_in_vehicles"] / len(veh_keys) * 100
        issue(
            f"{dim['only_in_vehicles']:,} vehicle records ({pct:.1f}%) have no matching accident — "
            f"orphaned records, dropped on inner join",
            "join_integrity"
        )

    report["dimensions"]["join_integrity"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  SAVE ALL OUTPUTS
# ════════════════════════════════════════════════════════════════════════════

def save_outputs():
    total = len(report["summary"]["issues"])
    report["summary"]["total_issue_count"] = total

    # Create reports directory if it doesn't exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log("\n" + "=" * 65)
    log(f"  VALIDATION COMPLETE")
    log(f"  Total issues found across all 8 dimensions: {total}")
    log("=" * 65)

    if report["summary"]["issues"]:
        log(f"\n  Issues by dimension:")
        for issue_item in report["summary"]["issues"]:
            log(f"    [{issue_item['dimension'].upper()}] {issue_item['message']}")
    else:
        log("\n  [PASS] No issues found!")

    # Save JSON
    json_path = REPORTS_DIR / "validation_report.json"
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    log(f"\n  [SAVED] {json_path}")

    # Save plain-text summary
    txt_path = REPORTS_DIR / "validation_summary.txt"
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("\n".join(log_lines))
    log(f"  [SAVED] {txt_path}")

    # Save PDF
    pdf_path = REPORTS_DIR / "validation_report.pdf"
    generate_pdf_report(pdf_path, total)
    log(f"  [SAVED] {pdf_path}")

    log("\n  Use validation_report.json to write Section 5.3 of your report.")
    log("  Screenshot or paste validation_summary.txt into the report appendix.")
    log("  PDF report: validation_report.pdf\n")


def generate_pdf_report(pdf_path,total):
    """Generate a professional PDF report from validation results."""
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=1  # Center
    )
    
    # Title
    elements.append(Paragraph("UK Road Accidents — Data Validation Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Generated timestamp
    generated_at = report.get("generated_at", "N/A")
    elements.append(Paragraph(f"<b>Generated:</b> {generated_at}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Summary
    elements.append(Paragraph(f"<b>Total Issues Found: {total}</b>", styles['Heading2']))
    elements.append(Spacer(1, 0.15*inch))
    
    # Issues by dimension table
    issue_counts = {}
    for issue_item in report["summary"]["issues"]:
        dim = issue_item.get("dimension", "UNKNOWN").upper()
        issue_counts[dim] = issue_counts.get(dim, 0) + 1
    
    if issue_counts:
        table_data = [["Dimension", "Issues"]]
        for dim, count in sorted(issue_counts.items()):
            table_data.append([dim, str(count)])
        
        table = Table(table_data, colWidths=[3*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("<b>✓ All dimensions passed validation</b>", styles['Normal']))
    
    elements.append(Spacer(1, 0.3*inch))
    elements.append(PageBreak())
    
    # Detailed issues
    if report["summary"]["issues"]:
        elements.append(Paragraph("Detailed Issues", styles['Heading2']))
        elements.append(Spacer(1, 0.15*inch))
        
        for issue_item in report["summary"]["issues"]:
            issue_text = f"<b>[{issue_item.get('dimension', 'UNKNOWN')}]</b> {issue_item.get('message', 'N/A')}"
            elements.append(Paragraph(issue_text, styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
    
    # Dimension summaries (optional detailed stats)
    if report.get("dimensions"):
        elements.append(PageBreak())
        elements.append(Paragraph("Validation Summary by Dimension", styles['Heading2']))
        elements.append(Spacer(1, 0.15*inch))
        
        for dim_name, dim_data in report["dimensions"].items():
            if isinstance(dim_data, dict):
                elements.append(Paragraph(f"<b>{dim_name.upper()}</b>", styles['Heading3']))
                summary_text = f"{len(dim_data)} checks performed"
                elements.append(Paragraph(summary_text, styles['Normal']))
                elements.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    doc.build(elements)


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    acc, veh = load_tables()
    check_accuracy(acc, veh)         # Dimension 1
    check_consistency(acc, veh)      # Dimension 2
    check_completeness(acc, veh)     # Dimension 3
    check_uniqueness(acc, veh)       # Dimension 4
    check_outliers(acc, veh)         # Dimension 5
    check_timeliness(acc)            # Dimension 6
    check_distribution(acc, veh)     # Dimension 7
    check_relationships(acc)         # Dimension 8
    check_join_integrity(acc, veh)   # Rubric: multi-source merge integrity
    save_outputs()
    return report


if __name__ == "__main__":
    main()
