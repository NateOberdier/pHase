value_cols = [c for c in df_raw.columns if c not in ["Date", "Time"]]
df_long = df_raw.melt(
    id_vars=["Date", "Time"],
    value_vars=value_cols,
    var_name="jar_id",
    value_name="pH"
)

df_long["datetime"] = pd.to_datetime(
    df_long["Date"].astype(str) + " " + df_long["Time"].astype(str),
    format="%d-%b %H:%M"
)

t0 = df_long["datetime"].min()
df_long["hours_since_milking"] = (df_long["datetime"] - t0).dt.total_seconds() / 3600.0
df_long["days_since_milking"] = df_long["hours_since_milking"] / 24.0
df_long["jar_num"] = df_long["jar_id"].astype(int)
df_long["has_culture"] = (df_long["jar_num"] > 6).astype(int)

def engineer_features(df_long):
    df = df_long.copy()
    df["pH_smooth"] = (
        df.groupby("jar_id")["pH"]
          .apply(lambda s: s.ewm(alpha=0.3, adjust=False).mean())
          .reset_index(level=0, drop=True)
    )
    def compute_slope(g):
        h = g["hours_since_milking"].values
        p = g["pH_smooth"].values
        dpdt = np.zeros_like(p)
        if len(p) > 1:
            dpdt[1:] = np.diff(p) / np.maximum(1e-3, np.diff(h))
        return np.clip(dpdt, -0.2, 0.2)
    df["dpH_dt"] = df.groupby("jar_id", group_keys=False).apply(
        lambda g: pd.Series(compute_slope(g), index=g.index)
    )
    df["temperature_F"] = CONFIG["temperature_F"]
    df["pH_x_hours"] = df["pH_smooth"] * df["hours_since_milking"]
    df["hours_sq"] = df["hours_since_milking"] ** 2
    return df

df_feat = engineer_features(df_long)

def heuristic_stage(ph, days):
    if days >= 30 and ph >= 5.0:
        return "cheese"
    if days >= 20 and ph >= 4.6:
        return "curding"
    if ph > 6.45:
        return "fresh"
    if ph >= 5.0:
        return "sour"
    if ph >= 4.6:
        return "yogurt"
    return "kefir"

def assign_stages_monotonic(df):
    df = df.sort_values(["jar_id", "hours_since_milking"]).copy()
    labels = []
    for jar, g in df.groupby("jar_id"):
        prev_idx = 0
        for _, row in g.iterrows():
            raw = heuristic_stage(row["pH_smooth"], row["days_since_milking"])
            idx_raw = STAGE_ORDER.index(raw)
            idx = max(prev_idx, idx_raw)
            labels.append(STAGE_ORDER[idx])
            prev_idx = idx
    df["stage_hard"] = labels
    return df

df_feat = assign_stages_monotonic(df_feat)

print("Feature data shape:", df_feat.shape)
print("Stage counts:")
print(df_feat["stage_hard"].value_counts().sort_index())

plt.figure(figsize=(10, 5))
for jar in sorted(df_feat["jar_id"].unique(), key=lambda x: int(x)):
    sub = df_feat[df_feat["jar_id"] == jar]
    plt.plot(sub["days_since_milking"], sub["pH"], marker="o", linestyle="-", label=f"Jar {jar}")
plt.xlabel("Days since milking")
plt.ylabel("pH")
plt.title("pH over time for all 12 jars")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
