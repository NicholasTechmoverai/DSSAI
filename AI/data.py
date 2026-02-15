import pandas as pd

data = pd.read_csv("AI/disease_prediction_dataset.csv")
print(f"Total Rows: {len(data)}")
print(f"Unique Diseases: {data['disease'].nunique()}")
print(f"\nFirst 5 rows:")
print(data.head())

# Save to CSV
# data.to_csv('disease_prediction_dataset.csv', index=False)
