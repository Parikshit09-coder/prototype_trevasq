import sys
import pickle
import pandas as pd
import json
from main import evaluate_model

model_path = sys.argv[1]
csv_path = sys.argv[2]
model_type = sys.argv[3] if len(sys.argv) > 3 else "QAUM"

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(csv_path)
    _, _, _, _, metrics, _ = evaluate_model(df, model, model_type)
    print(json.dumps(metrics))

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)