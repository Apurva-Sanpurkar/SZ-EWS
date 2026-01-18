import pandas as pd
import os

BASE = r"F:\VIT\SZEWS"

ENROL_PATH = os.path.join(BASE, "data_processed", "enrolment_all.csv")
DEMO_PATH  = os.path.join(BASE, "data_processed", "demographic_all.csv")
BIO_PATH   = os.path.join(BASE, "data_processed", "biometric_all.csv")

OUT_PATH = os.path.join(BASE, "data_processed", "SZEWS_timeseries_monthly.csv")


def standardize_common(df):
    df.columns = [c.strip().lower() for c in df.columns]

    # Required basic columns
    for col in ["date", "state", "district"]:
        if col not in df.columns:
            raise Exception(f"❌ Missing required column: {col}")

    # Auto-detect PIN column
    pin_candidates = ["pin_code", "pincode", "pin", "postal_code"]
    pin_col = None
    for c in pin_candidates:
        if c in df.columns:
            pin_col = c
            break
    if pin_col is None:
        raise Exception(f"❌ Missing PIN column. Tried {pin_candidates}. Found: {list(df.columns)}")

    df = df.rename(columns={pin_col: "pin_code"})

    # Date conversion
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Month key
    df["yyyymm"] = df["date"].dt.to_period("M").astype(str)

    # Clean geography
    df["state"] = df["state"].astype(str).str.strip()
    df["district"] = df["district"].astype(str).str.strip()

    # PIN safety
    df["pin_code"] = pd.to_numeric(df["pin_code"], errors="coerce")
    df = df.dropna(subset=["pin_code"])
    df["pin_code"] = df["pin_code"].astype(int).astype(str)

    # Region Key
    df["region_id"] = df["state"] + " | " + df["district"] + " | " + df["pin_code"]

    return df


# ---------------- ENROLMENT ----------------
print("✅ Loading enrolment...")
enrol = pd.read_csv(ENROL_PATH)
enrol = standardize_common(enrol)

enrol_parts = ["age_0_5", "age_5_17", "age_18_greater"]
for c in enrol_parts:
    if c not in enrol.columns:
        raise Exception(f"❌ Missing enrolment column: {c}")

enrol["enrol_activity"] = (
    pd.to_numeric(enrol["age_0_5"], errors="coerce").fillna(0)
    + pd.to_numeric(enrol["age_5_17"], errors="coerce").fillna(0)
    + pd.to_numeric(enrol["age_18_greater"], errors="coerce").fillna(0)
)

print("✅ Enrolment activity computed using:", enrol_parts)


# ---------------- DEMOGRAPHIC ----------------
print("✅ Loading demographic...")
demo = pd.read_csv(DEMO_PATH)
demo = standardize_common(demo)

demo_parts = ["demo_age_5_17", "demo_age_17_"]
for c in demo_parts:
    if c not in demo.columns:
        raise Exception(f"❌ Missing demographic column: {c}")

demo["demo_activity"] = (
    pd.to_numeric(demo["demo_age_5_17"], errors="coerce").fillna(0)
    + pd.to_numeric(demo["demo_age_17_"], errors="coerce").fillna(0)
)

print("✅ Demographic activity computed using:", demo_parts)


# ---------------- BIOMETRIC ----------------
print("✅ Loading biometric...")
bio = pd.read_csv(BIO_PATH)
bio = standardize_common(bio)

bio_parts = ["bio_age_5_17", "bio_age_17_"]
for c in bio_parts:
    if c not in bio.columns:
        raise Exception(f"❌ Missing biometric column: {c}")

bio["bio_activity"] = (
    pd.to_numeric(bio["bio_age_5_17"], errors="coerce").fillna(0)
    + pd.to_numeric(bio["bio_age_17_"], errors="coerce").fillna(0)
)

print("✅ Biometric activity computed using:", bio_parts)


# ---------------- MONTHLY AGGREGATION ----------------
print("✅ Aggregating monthly activity...")

enrol_m = enrol.groupby(["region_id", "state", "district", "pin_code", "yyyymm"], as_index=False).agg(
    enrol_activity=("enrol_activity", "sum")
)

demo_m = demo.groupby(["region_id", "state", "district", "pin_code", "yyyymm"], as_index=False).agg(
    demo_activity=("demo_activity", "sum")
)

bio_m = bio.groupby(["region_id", "state", "district", "pin_code", "yyyymm"], as_index=False).agg(
    bio_activity=("bio_activity", "sum")
)

print("✅ Merging datasets (outer join)...")
df = enrol_m.merge(demo_m, on=["region_id", "state", "district", "pin_code", "yyyymm"], how="outer")
df = df.merge(bio_m, on=["region_id", "state", "district", "pin_code", "yyyymm"], how="outer")

# Fill missing months with 0 activity
for col in ["enrol_activity", "demo_activity", "bio_activity"]:
    df[col] = df[col].fillna(0)

# Total Aadhaar service activity
df["total_activity"] = df["enrol_activity"] + df["demo_activity"] + df["bio_activity"]

# Sort for rolling baseline
df["yyyymm_dt"] = pd.to_datetime(df["yyyymm"] + "-01")
df = df.sort_values(["region_id", "yyyymm_dt"]).reset_index(drop=True)

# Rolling baseline (6-month moving avg)
print("✅ Computing baseline_total_ma6 (rolling mean)...")
df["baseline_total_ma6"] = df.groupby("region_id")["total_activity"].transform(
    lambda x: x.rolling(6, min_periods=3).mean()
)

# Suppression ratio = actual / baseline
df["suppression_ratio"] = df["total_activity"] / df["baseline_total_ma6"]
df["suppression_ratio"] = df["suppression_ratio"].replace([float("inf"), -float("inf")], 0).fillna(0)

# Save output
df.to_csv(OUT_PATH, index=False)

print("\n✅ STEP 3 DONE SUCCESSFULLY!")
print("✅ Saved file:", OUT_PATH)
print("✅ Rows:", len(df))
print("✅ Columns:", list(df.columns))
