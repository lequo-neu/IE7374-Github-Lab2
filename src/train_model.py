# src/train_model.py
# IE7374 Lab 2 — Advanced: Hyperparameter Sweep
# Changes from original:
#   - Model: RandomForestClassifier -> GradientBoostingClassifier
#   - Dataset: fixed n_samples=1000, n_features=8, n_informative=5
#   - Hyperparameter sweep: 6 configs compared on validation F1
#   - Best config selected automatically; final model trained on full dataset
#   - All sweep runs logged to MLflow for comparison

import argparse
import datetime
import os
import pickle
import sys

import mlflow
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath('..'))

SWEEP_CONFIGS = [
    {"n_estimators": 50,  "learning_rate": 0.10},
    {"n_estimators": 100, "learning_rate": 0.10},
    {"n_estimators": 200, "learning_rate": 0.10},
    {"n_estimators": 50,  "learning_rate": 0.05},
    {"n_estimators": 100, "learning_rate": 0.05},
    {"n_estimators": 200, "learning_rate": 0.05},
]

DATASET_NAME = "Drug Shortage Synthetic Dataset"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Timestamp: {timestamp}")

    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=42,
        shuffle=True,
    )

    os.makedirs('data', exist_ok=True)
    with open('data/data.pickle', 'wb') as f:
        pickle.dump(X, f)
    with open('data/target.pickle', 'wb') as f:
        pickle.dump(y, f)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_id = mlflow.create_experiment(f"{DATASET_NAME}_sweep_{current_time}")

    print(f"\n{'=' * 60}")
    print(f"Hyperparameter Sweep — {len(SWEEP_CONFIGS)} configurations")
    print(f"Train: {len(X_train)} samples | Validation: {len(X_val)} samples")
    print(f"{'=' * 60}")

    best_f1 = -1.0
    best_config = None
    sweep_results = []

    for i, config in enumerate(SWEEP_CONFIGS, start=1):
        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=f"sweep_config_{i}"):
            mlflow.log_params({
                "dataset_name": DATASET_NAME,
                "n_samples":    X.shape[0],
                "n_features":   X.shape[1],
                **config,
            })

            model = GradientBoostingClassifier(
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                random_state=42,
            )
            model.fit(X_train, y_train)

            y_pred_val = model.predict(X_val)
            val_f1 = f1_score(y_val, y_pred_val, average="weighted")
            mlflow.log_metric("val_f1", val_f1)

            is_best = val_f1 > best_f1
            if is_best:
                best_f1 = val_f1
                best_config = config

            sweep_results.append((config, val_f1))
            marker = " <-- best so far" if is_best else ""
            print(
                f"  Config {i}: "
                f"n_estimators={config['n_estimators']:>3}  "
                f"learning_rate={config['learning_rate']:.2f}  "
                f"| val F1: {val_f1:.4f}{marker}"
            )

    print(f"{'=' * 60}")
    print(
        f"  Best config: "
        f"n_estimators={best_config['n_estimators']}, "
        f"learning_rate={best_config['learning_rate']}  "
        f"(val F1: {best_f1:.4f})"
    )
    print(f"{'=' * 60}\n")

    with mlflow.start_run(experiment_id=experiment_id, run_name="final_model"):
        mlflow.log_params({
            "dataset_name":        DATASET_NAME,
            "n_samples":           X.shape[0],
            "n_features":          X.shape[1],
            "selected_by":         "sweep_best_val_f1",
            "sweep_best_val_f1":   best_f1,
            **best_config,
        })

        final_model = GradientBoostingClassifier(
            n_estimators=best_config["n_estimators"],
            learning_rate=best_config["learning_rate"],
            random_state=42,
        )
        final_model.fit(X, y)

        train_f1 = f1_score(y, final_model.predict(X), average="weighted")
        train_acc = accuracy_score(y, final_model.predict(X))
        mlflow.log_metrics({
            "train_f1":      train_f1,
            "train_acc":     train_acc,
            "sweep_val_f1":  best_f1,
        })

    os.makedirs('models', exist_ok=True)
    model_filename = f'model_{timestamp}_dt_model.joblib'
    dump(final_model, model_filename)
    print(f"Final model saved: {model_filename}")
    print(
        f"Config: n_estimators={best_config['n_estimators']}, "
        f"learning_rate={best_config['learning_rate']}"
    )
