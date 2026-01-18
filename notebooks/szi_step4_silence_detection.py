import pandas as pd
import os

BASE = r"F:\VIT\SZEWS"

IN_PATH = os.path.join(BASE, "data_processed", "SZEWS_timeseries_monthly.csv")
OUT_PATH = os.path.join(BASE, "data_processed", "SZEWS_final.csv")


def compute_consecutive_runs(flag_series):
    """
    Takes boolean series and returns consecutive run length count.
    Example: [0,1,1,0,1] -> [0,1,2,0,1]
    """
    count = 0
    out = []
    for v in flag_series:
        if v:
            count += 1
        else:
            count = 0
        out.append(count)
    return out


print("✅ Loading monthly time series...")
df = pd.read_csv(IN_PATH)

# Ensure sorting
df["yyyymm_dt"] = pd.to_datetime(df["yyyymm_dt"], errors="coerce")
df = df.sort_values(["region_id", "yyyymm_dt"]).reset_index(drop=True)

# ---------------- Silence Rules ----------------
# suppression_ratio = actual / baseline
# moderate: < 0.60
# severe:   < 0.40

df["flag_moderate"] = df["suppression_ratio"] < 0.60
df["flag_severe"]   = df["suppression_ratio"] < 0.40

# ---------------- Consecutive Durations ----------------
print("✅ Computing consecutive silence durations...")

df["moderate_run"] = df.groupby("region_id")["flag_moderate"].transform(compute_consecutive_runs)
df["severe_run"]   = df.groupby("region_id")["flag_severe"].transform(compute_consecutive_runs)

# silence label rules (as per problem statement):
# Moderate Silence: suppression < 60% for >= 2 consecutive months
# Severe Silence: suppression < 40% for >= 3 consecutive months

df["silence_state"] = "Normal"
df.loc[df["moderate_run"] >= 2, "silence_state"] = "Moderate"
df.loc[df["severe_run"] >= 3, "silence_state"] = "Severe"

# ---------------- Suppression Depth ----------------
# how much below baseline
df["suppression_depth_pct"] = (1 - df["suppression_ratio"]) * 100
df["suppression_depth_pct"] = df["suppression_depth_pct"].clip(lower=0)

# ---------------- Silence Duration Column ----------------
# final duration for reporting
df["silence_duration_months"] = 0
df.loc[df["silence_state"] == "Moderate", "silence_duration_months"] = df["moderate_run"]
df.loc[df["silence_state"] == "Severe", "silence_duration_months"] = df["severe_run"]

# ---------------- Silence Zone Index (SZI) ----------------
# SZI near 1 => normal activity
# SZI near 0 => severe silence
# We use suppression_ratio as base but cap 0..1
df["SZI"] = df["suppression_ratio"].clip(lower=0, upper=1)

# ---------------- Alert Logic (Early Warning) ----------------
# trigger alert if:
# - Severe now, OR
# - newly entered Moderate this month (run == 2), OR
# - newly entered Severe this month (run == 3)

df["alert_flag"] = 0
df.loc[df["severe_run"] == 3, "alert_flag"] = 1
df.loc[df["moderate_run"] == 2, "alert_flag"] = 1
df.loc[df["silence_state"] == "Severe", "alert_flag"] = 1

# ---------------- Recommendation Tag ----------------
def recommend(row):
    if row["silence_state"] == "Severe":
        # strong intervention
        if row["suppression_depth_pct"] > 70:
            return "Deploy Mobile Van + Temporary Camp + Infra Audit"
        return "Deploy Mobile Van + Temporary Camp"
    elif row["silence_state"] == "Moderate":
        return "Outreach + Temporary Camp + Monitoring"
    return "Normal Monitoring"

df["recommendation"] = df.apply(recommend, axis=1)

# ---------------- Final save ----------------
# Keep important columns for dashboard/app
final_cols = [
    "yyyymm", "yyyymm_dt",
    "state", "district", "pin_code", "region_id",
    "enrol_activity", "demo_activity", "bio_activity", "total_activity",
    "baseline_total_ma6",
    "suppression_ratio",
    "suppression_depth_pct",
    "moderate_run", "severe_run",
    "silence_duration_months",
    "SZI",
    "silence_state",
    "alert_flag",
    "recommendation"
]

df_final = df[final_cols].copy()
df_final.to_csv(OUT_PATH, index=False)

print("\n✅ STEP 4 DONE SUCCESSFULLY!")
print("✅ Saved:", OUT_PATH)
print("✅ Rows:", len(df_final))
print("✅ Severe count:", (df_final["silence_state"] == "Severe").sum())
print("✅ Moderate count:", (df_final["silence_state"] == "Moderate").sum())
