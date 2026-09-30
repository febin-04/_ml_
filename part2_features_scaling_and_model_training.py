# ---- PART 2 of 4 (run parts 1 to 4 in order, in the same Python session) ----

# ============================================================
# 14. CREATE MODELS
# ============================================================

knn_without_scaling = KNeighborsRegressor(
    n_neighbors=5
)

knn_with_scaling = KNeighborsRegressor(
    n_neighbors=5
)

lr_without_scaling = LinearRegression()

lr_with_scaling = LinearRegression()


# ============================================================
# 15. TRAIN MODELS
# ============================================================

knn_without_scaling.fit(
    X_train,
    y_train
)

knn_with_scaling.fit(
    X_train_scaled,
    y_train
)

lr_without_scaling.fit(
    X_train,
    y_train
)

lr_with_scaling.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 16. PREDICTIONS
# ============================================================

pred_knn_without = (
    knn_without_scaling.predict(X_test)
)

pred_knn_with = (
    knn_with_scaling.predict(X_test_scaled)
)

pred_lr_without = (
    lr_without_scaling.predict(X_test)
)

pred_lr_with = (
    lr_with_scaling.predict(X_test_scaled)
)


# ============================================================
# 17. EVALUATION FUNCTION
# ============================================================

def evaluate(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


# ============================================================
# 18. EVALUATE MODELS
# ============================================================

mae_knn_without, rmse_knn_without, r2_knn_without = evaluate(
    y_test,
    pred_knn_without
)

mae_knn_with, rmse_knn_with, r2_knn_with = evaluate(
    y_test,
    pred_knn_with
)

mae_lr_without, rmse_lr_without, r2_lr_without = evaluate(
    y_test,
    pred_lr_without
)

mae_lr_with, rmse_lr_with, r2_lr_with = evaluate(
    y_test,
    pred_lr_with
)


# ============================================================
# 19. COMPARISON TABLE
# ============================================================

results = pd.DataFrame({

    "Model": [
        "KNN",
        "KNN",
        "Linear Regression",
        "Linear Regression"
    ],

    "Scaling": [
        "Without Scaling",
        "With Scaling",
        "Without Scaling",
        "With Scaling"
    ],

    "MAE": [
        mae_knn_without,
        mae_knn_with,
        mae_lr_without,
        mae_lr_with
    ],

    "RMSE": [
        rmse_knn_without,
        rmse_knn_with,
        rmse_lr_without,
        rmse_lr_with
    ],

    "R2 Score": [
        r2_knn_without,
        r2_knn_with,
        r2_lr_without,
        r2_lr_with
    ]
})


print("\n\nMODEL COMPARISON")
print("=" * 100)

print(
    results.round(4).to_string(index=False)
)


# ============================================================
# 20. GRAPH - MONTHLY ELECTRICITY BILL
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly["Month"].astype(str),
    monthly["Monthly_Bill"],
    marker="o"
)

plt.title(
    "Monthly Electricity Bill"
)

plt.xlabel("Month")

plt.ylabel(
    "Electricity Bill (₹)"
)

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.show()
