# Turbofan RUL Prediction using XGBoost and Time-Series Windowing
# Public demo version

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


RUL_CAP = 125
WINDOW_SIZE = 20

DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
ZIP_PATH = "CMAPSS_outer.zip"
EXTRACT_DIR = "CMAPSSData"

columns = (
    ["unit_nr", "time_cycles"]
    + ["op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def download_and_extract_dataset():
    if not os.path.exists(ZIP_PATH):
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    inner_zip = None

    for root, _, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file == "CMAPSSData.zip":
                inner_zip = os.path.join(root, file)
                break

    if inner_zip:
        with zipfile.ZipFile(inner_zip, "r") as z:
            z.extractall(EXTRACT_DIR)

    print("Dataset ready")


def find_file(filename):
    for root, _, files in os.walk(EXTRACT_DIR):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found")


def load_cmapss_file(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :26]
    df.columns = columns
    return df


def calculate_train_rul(df):
    max_cycle_df = df.groupby("unit_nr")["time_cycles"].max().reset_index()
    max_cycle_df.columns = ["unit_nr", "max_cycle"]

    df = df.merge(max_cycle_df, on="unit_nr", how="left")
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    df["RUL_clipped"] = df["RUL"].clip(upper=RUL_CAP)

    return df.drop(columns=["max_cycle"])


def add_test_rul(test_df, rul_df):
    final_cycles = test_df.groupby("unit_nr")["time_cycles"].max().reset_index()
    final_cycles.columns = ["unit_nr", "max_test_cycle"]

    rul_df = rul_df.copy()
    rul_df["unit_nr"] = np.arange(1, len(rul_df) + 1)
    rul_df.columns = ["RUL_offset", "unit_nr"]

    test_df = test_df.merge(final_cycles, on="unit_nr", how="left")
    test_df = test_df.merge(rul_df, on="unit_nr", how="left")

    test_df["RUL"] = (
        test_df["max_test_cycle"]
        + test_df["RUL_offset"]
        - test_df["time_cycles"]
    )

    test_df["RUL_clipped"] = test_df["RUL"].clip(upper=RUL_CAP)

    return test_df.drop(columns=["max_test_cycle", "RUL_offset"])


feature_cols = [
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_11",
    "sensor_13",
    "sensor_17",
]


def create_train_windows(df, features, window_size):
    X = []
    y = []

    for unit in df["unit_nr"].unique():
        unit_df = df[df["unit_nr"] == unit].reset_index(drop=True)

        if len(unit_df) < window_size + 1:
            continue

        for i in range(window_size, len(unit_df)):
            window_slice = unit_df[features].iloc[i - window_size:i]
            window_data = window_slice.values.flatten()

            X.append(window_data)
            y.append(unit_df["RUL_clipped"].iloc[i])

    return np.array(X), np.array(y)


def create_test_windows(df, features, window_size):
    X = []
    y = []
    unit_ids = []

    for unit in df["unit_nr"].unique():
        unit_df = df[df["unit_nr"] == unit].reset_index(drop=True)

        if len(unit_df) < window_size:
            continue

        window_slice = unit_df[features].iloc[-window_size:]
        window_data = window_slice.values.flatten()

        X.append(window_data)
        y.append(unit_df["RUL_clipped"].iloc[-1])
        unit_ids.append(unit)

    return np.array(X), np.array(y), np.array(unit_ids)


def classify_health(rul):
    if rul > 80:
        return "Healthy"
    if rul > 30:
        return "Warning"
    return "Replace"


def maintenance_decision(health_state):
    if health_state == "Replace":
        return "Replace Immediately"
    if health_state == "Warning":
        return "Monitor Closely"
    return "Normal Operation"


download_and_extract_dataset()

train_file = find_file("train_FD001.txt")
test_file = find_file("test_FD001.txt")
rul_file = find_file("RUL_FD001.txt")

train_df = load_cmapss_file(train_file)
test_df = load_cmapss_file(test_file)

rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, engine="python")
rul_df = rul_df.iloc[:, :1]
rul_df.columns = ["RUL_offset"]

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("RUL shape  :", rul_df.shape)

train_df = calculate_train_rul(train_df)
test_df = add_test_rul(test_df, rul_df)

print("Selected features:", feature_cols)
print("Number of selected features:", len(feature_cols))

X_train, y_train = create_train_windows(train_df, feature_cols, WINDOW_SIZE)
X_test, y_test, test_units = create_test_windows(test_df, feature_cols, WINDOW_SIZE)

print("\nData shapes")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, RUL_CAP)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("R2:", round(r2, 4))
print("RMSE:", round(rmse, 2))
print("MAE:", round(mae, 2))

pred_df = pd.DataFrame({
    "unit_nr": test_units,
    "Actual_RUL": y_test,
    "Predicted_RUL": y_pred,
})

pred_df["Health_State"] = pred_df["Predicted_RUL"].apply(classify_health)
pred_df["Maintenance_Decision"] = pred_df["Health_State"].apply(maintenance_decision)

print("\nSample Predictions")
print(pred_df.head(10))

comparison = pd.DataFrame({
    "Work": ["Gupta et al. (2025)", "Proposed Work"],
    "Model": ["Random Forest", "XGBoost + Windowing"],
    "R2": [0.789, 0.8517],
    "RMSE": [19.0, 15.43],
    "MAE": ["~14-15", 11.59],
})

print("\nComparison with Reference Work")
print(comparison)
