import joblib
import pandas as pd
import time
model = joblib.load("../kidney_disease_model.pkl")
df = pd.read_csv("../kidney_disease.csv")
for i in range(10):
    row = df.iloc[[i]]
    X = row.drop(columns=["classification", "id"])

    prediction = model.predict(X)

    print(f"Event {i + 1}")
    print(f"ID: {row['id'].values[0]}")
    print(f"Prediction: {prediction[0]}")
    print("-" * 30)

    time.sleep(1)
