# training/validate_data.py

import pandas as pd
import sys

DATA_PATH = "data/clean.csv"

df = pd.read_csv(DATA_PATH)

# 1. No empty dataframe
if df.shape[0] == 0:
    print("❌ Empty dataset")
    sys.exit(1)

# 2. No missing values
missing = df.isnull().sum().sum()
if missing > 0:
    print("❌ Missing values detected")
    sys.exit(1)

# 3. Target exists
if "Churn" not in df.columns:
    print("❌ Target column missing")
    sys.exit(1)

print("✅ Data validation passed")
