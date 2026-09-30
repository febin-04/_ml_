# Impact of Feature Scaling on KNN and Linear Regression for Electricity Bill Prediction

An end-to-end Machine Learning project studying how feature scaling affects distance-based algorithms (**K-Nearest Neighbors**) versus equation-based algorithms (**Linear Regression**) when predicting monthly household electricity bills.

---

## 📌 Project Overview & Abstract

- **Dataset:** [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (~2.07 million records).
- **Tariff Assumed:** ₹7 per kWh.
- **Target Variable:** `Monthly_Bill` (derived from monthly aggregated `Energy_kWh × ₹7`).
- **Features (7):** `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`.
- **Core Finding:** Feature scaling (`StandardScaler`) significantly boosts KNN accuracy ($R^2$ jumps from **0.6860 to 0.8393**), while Linear Regression naturally adapts ($R^2 \approx 0.925$–$0.928$) regardless of feature scales.

---

## 📁 Project Structure

| File | Description |
| :--- | :--- |
| **`part1_data_loading_and_preparation.py`** | Loads raw 2M+ records, handles missing values (`?`), resamples minute-level data into **48 monthly periods**, computes bill amounts, and splits data (80% train / 20% test). |
| **`part2_features_scaling_and_model_training.py`** | Fits **StandardScaler**, trains 4 model variations (KNN and Linear Regression with/without scaling), predicts on test set, and displays comparison metrics. |
| **`part3_evaluation_and_graphs.py`** | Generates comparison bar charts (MAE, RMSE, $R^2$), scatter plots (Actual vs Predicted), and exports `monthly_electricity_bill.csv`. |
| **`part4_user_input_and_prediction.py`** | Interactive console tool allowing users to enter custom household metrics and obtain predicted monthly bills. |
| **`run_all.py`** | Master script to execute all four parts sequentially in a single session. |

---

## 📊 Experimental Results

| Model | Scaling Status | MAE (₹) | RMSE (₹) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **KNN ($k=5$)** | Without Scaling | 673.05 | 830.45 | 0.6860 |
| **KNN ($k=5$)** | **With Scaling** | **475.18** | **594.19** | **0.8393** |
| **Linear Regression** | Without Scaling | 298.92 | 397.03 | **0.9282** |
| **Linear Regression** | With Scaling | 309.26 | 406.86 | 0.9246 |

> **Takeaway:** Distance metrics (Euclidean) in KNN get overwhelmed by high-magnitude features (e.g. Sub-metering Wh at ~275,000) over low-magnitude features (e.g. Active Power at ~1 kW). StandardScaler normalizes all features to mean 0 and variance 1, restoring balanced distance weights.

---

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. **Execute all parts end-to-end:**
   ```bash
   python run_all.py
   ```
   *(Ensure `household_power_consumption.txt` is present in the project directory).*