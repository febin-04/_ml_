# ---- PART 3 of 4 (run parts 1 to 4 in order, in the same Python session) ----

# ============================================================
# 21. GRAPH - MAE COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results,
    x="Model",
    y="MAE",
    hue="Scaling"
)

plt.title(
    "MAE Comparison"
)

plt.xlabel("Model")

plt.ylabel("MAE")

plt.tight_layout()

plt.show()


# ============================================================
# 22. GRAPH - RMSE COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results,
    x="Model",
    y="RMSE",
    hue="Scaling"
)

plt.title(
    "RMSE Comparison"
)

plt.xlabel("Model")

plt.ylabel("RMSE")

plt.tight_layout()

plt.show()


# ============================================================
# 23. GRAPH - R2 COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results,
    x="Model",
    y="R2 Score",
    hue="Scaling"
)

plt.title(
    "R² Score Comparison"
)

plt.xlabel("Model")

plt.ylabel("R² Score")

plt.tight_layout()

plt.show()


# ============================================================
# 24. ACTUAL VS PREDICTED - KNN
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    pred_knn_without,
    alpha=0.7,
    label="Without Scaling"
)

plt.scatter(
    y_test,
    pred_knn_with,
    alpha=0.7,
    label="With Scaling"
)

plt.xlabel(
    "Actual Monthly Bill (₹)"
)

plt.ylabel(
    "Predicted Monthly Bill (₹)"
)

plt.title(
    "KNN: Actual vs Predicted Monthly Bill"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 25. ACTUAL VS PREDICTED - LINEAR REGRESSION
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    pred_lr_without,
    alpha=0.7,
    label="Without Scaling"
)

plt.scatter(
    y_test,
    pred_lr_with,
    alpha=0.7,
    label="With Scaling"
)

plt.xlabel(
    "Actual Monthly Bill (₹)"
)

plt.ylabel(
    "Predicted Monthly Bill (₹)"
)

plt.title(
    "Linear Regression: Actual vs Predicted"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 26. SAVE MONTHLY DATASET
# ============================================================

monthly.to_csv(
    "monthly_electricity_bill.csv",
    index=False
)

print("\nMonthly dataset saved as:")
print("monthly_electricity_bill.csv")


# ============================================================
# 27. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("PROJECT COMPLETED")
print("=" * 70)

print("""
The project compares:

1. KNN without feature scaling
2. KNN with feature scaling
3. Linear Regression without feature scaling
4. Linear Regression with feature scaling

Evaluation metrics:
- MAE
- RMSE
- R² Score

The electricity bill is calculated using:

Monthly Bill = Monthly Energy Consumption × Tariff

Assumed tariff = ₹7 per kWh
""")
