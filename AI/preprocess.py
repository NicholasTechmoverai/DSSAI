from pandas import pd
DATA_PATH = "disease_prediction_dataset.csv"

try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} rows from '{DATA_PATH}'")
except Exception as e:
    raise RuntimeError("❌ Failed to load dataset") from e
