no_culture_jars = [1, 2, 3, 4, 5, 6]
with_culture_jars = [7, 8, 9, 10, 11, 12]

def stratify_group(jars, seed):
    np.random.seed(seed)
    jars = np.array(jars)
    np.random.shuffle(jars)
    train = jars[:4]
    val   = jars[4:5]
    test  = jars[5:6]
    return train.tolist(), val.tolist(), test.tolist()

no_train, no_val, no_test = stratify_group(no_culture_jars, CONFIG["random_seed"])
yes_train, yes_val, yes_test = stratify_group(with_culture_jars, CONFIG["random_seed"] + 1)

train_jars = no_train + yes_train
val_jars   = no_val   + yes_val
test_jars  = no_test  + yes_test

train_df = df_feat[df_feat["jar_num"].isin(train_jars)].copy()
val_df   = df_feat[df_feat["jar_num"].isin(val_jars)].copy()
test_df  = df_feat[df_feat["jar_num"].isin(test_jars)].copy()

print("Train jars:", sorted(train_df["jar_id"].unique(), key=int))
print("Val jars:  ", sorted(val_df["jar_id"].unique(), key=int))
print("Test jars: ", sorted(test_df["jar_id"].unique(), key=int))

FEATURES_FULL = [
    "pH_smooth",
    "hours_since_milking",
    "dpH_dt",
    "pH_x_hours",
    "hours_sq",
    "temperature_F",
    "has_culture"
]

X_train_full, y_train_full = train_df[FEATURES_FULL], train_df["stage_hard"]
X_val_full,   y_val_full   = val_df[FEATURES_FULL],   val_df["stage_hard"]
X_test_full,  y_test_full  = test_df[FEATURES_FULL],  test_df["stage_hard"]

clf_full = RandomForestClassifier(
    n_estimators=350,
    max_depth=None,
    random_state=CONFIG["random_seed"],
    n_jobs=-1
)
clf_full.fit(X_train_full, y_train_full)

y_pred_full = clf_full.predict(X_test_full)
proba_full = clf_full.predict_proba(X_test_full)

print("Random Forest stage classification (full features)")
print("Accuracy:", accuracy_score(y_test_full, y_pred_full))
print("Macro F1:", f1_score(y_test_full, y_pred_full, average="macro"))
print(classification_report(y_test_full, y_pred_full, digits=3))

stage_to_idx_full = {c: i for i, c in enumerate(clf_full.classes_)}
briers = []
for cls in clf_full.classes_:
    y_true = (y_test_full.values == cls).astype(int)
    y_hat  = proba_full[:, stage_to_idx_full[cls]]
    briers.append(brier_score_loss(y_true, y_hat))
print("Avg Brier score:", np.mean(briers))

reg_full = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    random_state=CONFIG["random_seed"],
    n_jobs=-1
)
reg_full.fit(train_df[FEATURES_FULL], train_df["pH"])

pH_hat_full = reg_full.predict(test_df[FEATURES_FULL])
mse_full  = mean_squared_error(test_df["pH"], pH_hat_full)
rmse_full = mse_full ** 0.5
r2_full   = r2_score(test_df["pH"], pH_hat_full)
print("Random Forest pH regression (full features)")
print("RMSE:", rmse_full)
print("R^2:", r2_full)

sample_jar = sorted(test_df["jar_id"].unique(), key=int)[0]
mask = test_df["jar_id"] == sample_jar
sub_true = test_df[mask].sort_values("hours_since_milking")
sub_pred = pH_hat_full[mask.values]

plt.figure(figsize=(8, 4))
plt.plot(sub_true["days_since_milking"], sub_true["pH"], "o-", label="Actual pH")
plt.plot(sub_true["days_since_milking"], sub_pred, "s--", label="Predicted pH")
plt.xlabel("Days since milking")
plt.ylabel("pH")
plt.title(f"pH prediction – Jar {sample_jar}")
plt.legend()
plt.tight_layout()
plt.show()

X_time_all = df_feat[["hours_since_milking", "has_culture"]]
y_pH_all = df_feat["pH"]

reg_time = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    random_state=CONFIG["random_seed"],
    n_jobs=-1
)
reg_time.fit(X_time_all, y_pH_all)

MAX_HOURS_DATA = float(df_feat["hours_since_milking"].max())
print("Max hours in dataset:", MAX_HOURS_DATA)

df_first = (
    df_feat.sort_values("hours_since_milking")
          .groupby(["has_culture", "stage_hard"])["hours_since_milking"]
          .min()
          .reset_index()
)

stage_first_times_cultured = {
    row["stage_hard"]: float(row["hours_since_milking"])
    for _, row in df_first[df_first["has_culture"] == 1].iterrows()
}

stage_first_times_noculture = {
    row["stage_hard"]: float(row["hours_since_milking"])
    for _, row in df_first[df_first["has_culture"] == 0].iterrows()
}

print("Label-derived first stage times (days) – cultured:")
for st in STAGE_ORDER:
    if st in stage_first_times_cultured:
        print(st, ":", stage_first_times_cultured[st] / 24.0)
    else:
        print(st, ": no data")

print("Label-derived first stage times (days) – no culture:")
for st in STAGE_ORDER:
    if st in stage_first_times_noculture:
        print(st, ":", stage_first_times_noculture[st] / 24.0)
    else:
        print(st, ": no data")
