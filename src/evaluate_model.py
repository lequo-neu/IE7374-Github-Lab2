# src/evaluate_model.py
import os
import sys
import json
import pickle
import argparse

import joblib
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp

    # Load the same data that was used during training
    with open('data/data.pickle', 'rb') as f:
        X = pickle.load(f)
    with open('data/target.pickle', 'rb') as f:
        y = pickle.load(f)

    # Hold out 20% as a consistent test split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    model_version = f'model_{timestamp}_dt_model'
    model = joblib.load(f'{model_version}.joblib')

    # Evaluate on held-out test split
    y_pred = model.predict(X_test)
    metrics = {"F1_Score": f1_score(y_test, y_pred, average='weighted')}

    print(f"F1 Score on test split: {metrics['F1_Score']:.4f}")

    # Save metrics
    os.makedirs('metrics', exist_ok=True)
    metrics_filename = f'{timestamp}_metrics.json'
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved: {metrics_filename}")
