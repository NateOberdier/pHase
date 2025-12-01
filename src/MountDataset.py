import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error, r2_score, brier_score_loss

warnings.filterwarnings("ignore")

FILE_ID = "1N72GfA76iHDD2AKpoevpY3hLpWH7Zszi"
OUTPUT_CSV = "MilkLog.csv"

if not os.path.exists(OUTPUT_CSV):
    import gdown
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", OUTPUT_CSV, quiet=False)

df_raw = pd.read_csv(OUTPUT_CSV)

STAGE_ORDER = [
    "fresh",
    "sour",
    "yogurt",
    "kefir",
    "curding",
    "cheese"
]

CONFIG = {
    "temperature_F": 68.0,
    "random_seed": 42
}

np.random.seed(CONFIG["random_seed"])

print("Raw data shape:", df_raw.shape)
print("Stages:", STAGE_ORDER)
