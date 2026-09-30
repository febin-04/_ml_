# ---- PART 1 of 4 (run parts 1 to 4 in order, in the same Python session) ----

# ============================================================
# IMPACT OF FEATURE SCALING ON KNN AND LINEAR REGRESSION
# FOR MONTHLY ELECTRICITY BILL PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "household_power_consumption.txt",
    sep=";",
    na_values="?",
    low_memory=False
)

print("\nORIGINAL DATASET")
print("=" * 70)

print(df.head(10).to_string())

print("\nDataset Shape:", df.shape)


# ============================================================
# 2. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 3. REMOVE MISSING VALUES
# ============================================================

df = df.dropna().copy()


# ============================================================
# 4. CREATE DATE COLUMN
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True
)


# ============================================================
# 5. CALCULATE ENERGY CONSUMPTION
# ============================================================

# Global_active_power = kW
# Each record represents approximately 1 minute.
#
# Energy used in one minute:
# kWh = kW / 60

df["Energy_kWh"] = (
    df["Global_active_power"] / 60
)


# ============================================================
# 6. CREATE MONTH COLUMN
# ============================================================

df["Month"] = df["Date"].dt.to_period("M")


# ============================================================
# 7. CREATE MONTHLY DATASET
# ============================================================

monthly = df.groupby("Month").agg({

    "Global_active_power": "mean",
    "Global_reactive_power": "mean",
    "Voltage": "mean",
    "Global_intensity": "mean",
    "Sub_metering_1": "sum",
    "Sub_metering_2": "sum",
    "Sub_metering_3": "sum",
    "Energy_kWh": "sum"

}).reset_index()


# ============================================================
# 8. CALCULATE MONTHLY ELECTRICITY BILL
# ============================================================

# Assumed electricity tariff
TARIFF = 7       # ₹7 per kWh

monthly["Monthly_Bill"] = (
    monthly["Energy_kWh"] * TARIFF
)


# ============================================================
# 9. DISPLAY MONTHLY DATASET AS TABLE
# ============================================================

print("\nMONTHLY DATASET")
print("=" * 100)

print(
    monthly.round(2).to_string(index=False)
)


# ============================================================
# 10. DISPLAY DATASET USING PANDAS TABLE
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

print("\nFirst 10 Monthly Records:")
print(
    monthly.head(10).round(2)
)


# ============================================================
# 11. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

X = monthly[features]

y = monthly["Monthly_Bill"]


# ============================================================
# 12. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 13. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)
