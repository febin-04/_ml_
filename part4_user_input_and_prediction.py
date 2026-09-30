# ---- PART 4 of 4 (run parts 1 to 4 in order, in the same Python session) ----

# ============================================================
# 28. USER INPUT - ACTUAL ELECTRICITY BILL PREDICTION
# ============================================================

print("\n")
print("=" * 70)
print("ELECTRICITY BILL PREDICTION SYSTEM")
print("=" * 70)

print("""
Enter the monthly household electricity parameters below.

The values should represent the same type of measurements
used in the training dataset.
""")


# ------------------------------------------------------------
# GET USER INPUT
# ------------------------------------------------------------

global_active_power = float(
    input("\nAverage Global Active Power (kW): ")
)

global_reactive_power = float(
    input("Average Global Reactive Power (kW): ")
)

voltage = float(
    input("Average Voltage (V): ")
)

global_intensity = float(
    input("Average Global Intensity (A): ")
)

sub_metering_1 = float(
    input("Sub Metering 1 (Wh): ")
)

sub_metering_2 = float(
    input("Sub Metering 2 (Wh): ")
)

sub_metering_3 = float(
    input("Sub Metering 3 (Wh): ")
)


# ============================================================
# 29. CREATE INPUT DATAFRAME
# ============================================================

user_input = pd.DataFrame({

    "Global_active_power": [
        global_active_power
    ],

    "Global_reactive_power": [
        global_reactive_power
    ],

    "Voltage": [
        voltage
    ],

    "Global_intensity": [
        global_intensity
    ],

    "Sub_metering_1": [
        sub_metering_1
    ],

    "Sub_metering_2": [
        sub_metering_2
    ],

    "Sub_metering_3": [
        sub_metering_3
    ]

})


# ============================================================
# 30. SCALE USER INPUT
# ============================================================

user_input_scaled = scaler.transform(
    user_input
)


# ============================================================
# 31. PREDICT ELECTRICITY BILL
# ============================================================

predicted_knn_without = (
    knn_without_scaling.predict(
        user_input
    )[0]
)

predicted_knn_with = (
    knn_with_scaling.predict(
        user_input_scaled
    )[0]
)

predicted_lr_without = (
    lr_without_scaling.predict(
        user_input
    )[0]
)

predicted_lr_with = (
    lr_with_scaling.predict(
        user_input_scaled
    )[0]
)


# ============================================================
# 32. DISPLAY PREDICTIONS
# ============================================================

print("\n")
print("=" * 70)
print("PREDICTED MONTHLY ELECTRICITY BILL")
print("=" * 70)

print(
    f"\nKNN Without Scaling      : ₹{predicted_knn_without:.2f}"
)

print(
    f"KNN With Scaling         : ₹{predicted_knn_with:.2f}"
)

print(
    f"Linear Regression Without Scaling : "
    f"₹{predicted_lr_without:.2f}"
)

print(
    f"Linear Regression With Scaling    : "
    f"₹{predicted_lr_with:.2f}"
)


# ============================================================
# 33. FINAL PREDICTION
# ============================================================

# Use the scaled Linear Regression model as the
# primary prediction model.

final_prediction = predicted_lr_with

# Prevent negative bill prediction
final_prediction = max(
    0,
    final_prediction
)


print("\n")
print("=" * 70)
print("FINAL ELECTRICITY BILL PREDICTION")
print("=" * 70)

print(
    f"\nEstimated Monthly Electricity Bill: "
    f"₹{final_prediction:.2f}"
)

print(
    "\nApproximate Bill: "
    f"₹{round(final_prediction)}"
)

print("\n")
print("=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)
