import joblib
import pandas as pd
import random

MODEL_PATH = "models/v1/model.pkl"

model = joblib.load(MODEL_PATH)

# Fake input (sample from training data)
df = pd.read_csv("data/clean.csv")
sample = df.drop("Churn_Yes", axis=1).sample(5)

preds = model.predict_proba(sample)[:, 1]

for i, p in enumerate(preds):
    print(f"Prediction {i+1}: churn_probability={p:.3f}")
