# training/train.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from datetime import datetime
import json
import os
import yaml

with open("training/config.yaml") as f:
    config = yaml.safe_load(f)

# Paths
DATA_PATH = "data/clean.csv"
MODEL_DIR = "models/v1"
MODEL_PATH = f"{MODEL_DIR}/model.pkl"
META_PATH = f"{MODEL_DIR}/metadata.json"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["data"]["test_size"],
    random_state=42
)


# Train
model = RandomForestClassifier(
    n_estimators=config["model"]["n_estimators"],
    random_state=config["model"]["random_state"]
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)

# Save model
joblib.dump(model, MODEL_PATH)

# Save reference data (IMPORTANT FOR DRIFT)
X_train.to_csv(f"{MODEL_DIR}/reference.csv", index=False)

# Save metadata
metadata = {
    "model_type": "RandomForest",
    "version": "v1",
    "auc": auc,
    "train_date": str(datetime.utcnow()),
    "features": list(X.columns)
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("Training complete")
print("AUC:", auc)
