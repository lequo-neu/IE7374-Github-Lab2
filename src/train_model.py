# src/train_model.py
# IE7374 Lab 2 — Modified from original train_model.py
# Changes:
#   - Model: RandomForestClassifier → GradientBoostingClassifier
#   - Dataset name: "Reuters Corpus Volume" → "Drug Shortage Synthetic Dataset"
#   - n_samples: randint(0, 2000) → randint(500, 2000)  [fix: tránh 0 samples]
#   - n_features: 6 → 8, n_informative: 3 → 5
#   - Added: log learning_rate and n_estimators as MLflow params

import mlflow, datetime, os, pickle, random
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import argparse, sys

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Timestamp received: {timestamp}")

    # --- Dataset (modified) ---
    X, y = make_classification(
        n_samples=random.randint(500, 2000),   # fix: was randint(0, 2000)
        n_features=8,                           # modified: was 6
        n_informative=5,                        # modified: was 3
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=42,
        shuffle=True,
    )

    # Save data
    os.makedirs('data', exist_ok=True)
    with open('data/data.pickle', 'wb') as f:
        pickle.dump(X, f)
    with open('data/target.pickle', 'wb') as f:
        pickle.dump(y, f)

    # --- MLflow tracking ---
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Drug Shortage Synthetic Dataset"   # modified
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_id = mlflow.create_experiment(f"{dataset_name}_{current_time}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=dataset_name):

        # Modified model: GradientBoostingClassifier
        n_estimators = 100
        learning_rate = 0.1

        mlflow.log_params({
            "dataset_name":      dataset_name,
            "n_samples":         X.shape[0],
            "n_features":        X.shape[1],
            "n_estimators":      n_estimators,    # added
            "learning_rate":     learning_rate,   # added
        })

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)

        mlflow.log_metrics({
            'Accuracy': accuracy_score(y, y_pred),
            'F1 Score': f1_score(y, y_pred),
        })

    # Save model with timestamp-based versioning
    os.makedirs('models', exist_ok=True)
    model_filename = f'model_{timestamp}_dt_model.joblib'
    dump(model, model_filename)
    print(f"Model saved: {model_filename}")