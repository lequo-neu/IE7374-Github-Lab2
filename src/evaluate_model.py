# src/evaluate_model.py
import pickle, os, json, random
from sklearn.metrics import f1_score
import joblib, sys
import argparse
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp

    # Load model
    model_version = f'model_{timestamp}_dt_model'
    model = joblib.load(f'{model_version}.joblib')

    # Generate evaluation data
    X, y = make_classification(
        n_samples=random.randint(500, 2000),
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=42,
        shuffle=True,
    )

    y_pred = model.predict(X)
    metrics = {"F1_Score": f1_score(y, y_pred)}

    # Save metrics
    os.makedirs('metrics', exist_ok=True)
    metrics_filename = f'{timestamp}_metrics.json'
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved: {metrics_filename}")