import joblib
import pandas as pd

model = joblib.load("../kidney_disease_model.pkl")

df = pd.read_csv("../kidney_disease.csv")

ids = df["id"]

X = df.drop(columns=["classification", "id"])

predictions = model.predict(X)

output = pd.DataFrame({
    "id": ids,
    "prediction": predictions
})

output.to_csv("batch_predictions.csv", index=False)

print("Batch inference completed")
print("\nPredictions:")
print(output.head())

print("\nOutput saved as: batch_predictions.csv")
