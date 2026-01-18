import pandas as pd
import glob
import os

BASE = r"F:\VIT\SZEWS"

def merge_csv_parts(folder_path, output_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not files:
        print("❌ No CSV files found in:", folder_path)
        return

    print(f"\n📌 Merging {len(files)} files from: {folder_path}")
    df_list = []
    for f in files:
        try:
            temp = pd.read_csv(f)
            df_list.append(temp)
            print("✅ Loaded:", os.path.basename(f), "| rows:", len(temp))
        except Exception as e:
            print("❌ Failed:", f, "|", e)

    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(output_path, index=False)
    print("✅ Saved merged file:", output_path)
    print("✅ Final rows:", len(df))


merge_csv_parts(
    folder_path=os.path.join(BASE, "data_raw", "enrolment"),
    output_path=os.path.join(BASE, "data_processed", "enrolment_all.csv")
)

merge_csv_parts(
    folder_path=os.path.join(BASE, "data_raw", "demographic"),
    output_path=os.path.join(BASE, "data_processed", "demographic_all.csv")
)

merge_csv_parts(
    folder_path=os.path.join(BASE, "data_raw", "biometric"),
    output_path=os.path.join(BASE, "data_processed", "biometric_all.csv")
)
