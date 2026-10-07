# IE7374 Lab 2 — Automated ML Model Training and Versioning

## Overview

This lab implements an automated machine learning pipeline using GitHub Actions.
On every push to main, the pipeline runs a hyperparameter sweep across six
GradientBoostingClassifier configurations, selects the best one by validation
F1 score, trains the final model on the full dataset, saves it to the repository
with a timestamp-based filename, and gates the commit on a minimum quality
threshold. A second workflow runs the same pipeline on a daily schedule. Every
run is fully logged to MLflow so the sweep results, the winning configuration,
and the final model metrics are all traceable.

## Changes Made to the Lab

**Source code (src/train_model.py)**

The original lab used RandomForestClassifier on a randomly-sized synthetic
dataset. The model was changed to GradientBoostingClassifier and the dataset
was fixed at 1000 samples with random_state=42. The dataset name was changed
to Drug Shortage Synthetic Dataset to reflect the PharmTrack Sentinel domain.
The MLflow tracking URI was migrated from the deprecated file store to SQLite.

**Pipeline (model_retraining_on_push.yml and model_calibration.yml)**

A model quality gate was added between the evaluate step and the commit step.
After evaluation, the pipeline reads the F1 score from the saved metrics JSON
and checks it against a threshold of 0.70. If the score is below the threshold,
the pipeline exits with an error and skips the commit, preventing a poor model
from being versioned into the repository.

## Advanced Extension — Hyperparameter Sweep with Automatic Best-Config Selection

**Why this was chosen**

Training a model with a single fixed set of hyperparameters and shipping it
directly is one of the most common sources of underperforming ML systems. In
practice, the right combination of learning rate and tree depth depends on the
specific dataset, and what works on one run may not generalize to the next batch
of data. In an MLOps pipeline that retrains automatically on a schedule, this
problem compounds: a configuration that was optimal at deployment time can
become suboptimal as the data distribution shifts, yet the pipeline keeps
retraining with the same settings and nobody notices.

A hyperparameter sweep solves this by treating configuration selection as part
of the training job itself rather than a one-time manual decision. It was chosen
over a single-model approach specifically because this lab's pipeline already
retrained on every push and on a daily schedule. Adding a sweep to that loop
means the pipeline is not just automating training but actively finding the best
version of the model each time it runs. The sweep also integrates naturally with
the existing MLflow logging, so every configuration tried in every run is
permanently recorded and comparable across runs.

GradientBoostingClassifier was selected as the base model because its two most
influential hyperparameters, n_estimators and learning_rate, have a well-known
trade-off: more trees with a smaller learning rate produces a stronger but slower
model, while fewer trees with a higher learning rate trains faster but risks
under-fitting. Testing both dimensions systematically is a textbook grid search
that illustrates the sweep concept clearly without obscuring it in complexity.

**How it works**

The sweep defines six configurations as the cross product of n_estimators in
{50, 100, 200} and learning_rate in {0.05, 0.10}. At the start of each pipeline
run, the full dataset is split 80/20. Each of the six configurations trains on
the 80% portion and evaluates on the 20% validation set, producing a weighted
F1 score. This split is held constant across all six configurations, which means
the comparison is fair: every configuration sees exactly the same training data
and is tested on exactly the same held-out samples.

All six runs are logged to MLflow as separate runs within the same experiment.
Each run records its hyperparameters and its validation F1 score, so the full
sweep result is always retrievable. After the sweep, the configuration with the
highest validation F1 is selected automatically without any human intervention.
The final model is then retrained on the full 100% of the data using that winning
configuration, which gives it slightly more signal than the validation-split
model. The final model and its MLflow run are both tagged with selected_by:
sweep_best_val_f1 so it is always clear how the configuration was chosen.

The dataset is fixed at 1000 samples with random_state=42 rather than a random
size. This is a deliberate design choice: if the sample count changed every run,
differences in validation F1 between configurations could reflect sampling
noise rather than genuine hyperparameter effects, making the sweep misleading.
With a fixed dataset, every difference in F1 is attributable to the
hyperparameters being compared.

**What it produces**

Every pipeline run produces a complete record of what was tried and why the
winning model was chosen. The MLflow experiment view shows all six sweep runs
side by side with their F1 scores, making it trivial to compare learning rates
or tree counts across runs. The console output prints a formatted table during
the GitHub Actions run, so anyone reading the log can see at a glance which
configurations were tested, what scores they achieved, and which one was
selected. This level of transparency is what separates a reproducible ML system
from a black box that happens to produce a model file.

## Prerequisites

The following must be available on your machine before running locally.

Python 3.14 or later. Download from https://www.python.org/downloads and
verify with: python3 --version

Git. Download from https://git-scm.com and verify with: git --version

## Environment Setup

Clone the repository.

```
git clone https://github.com/lequo-neu/IE7374-Github-Lab2.git
cd IE7374-Github-Lab2
```

Create and activate a virtual environment.

```
python3 -m venv lab_env
source lab_env/bin/activate
```

Install all dependencies.

```
pip install -r requirements.txt
```

Create the output folders if they do not already exist.

```
mkdir -p models metrics data
```

## Running the Code Locally

Generate a timestamp and run the sweep.

```
timestamp=$(date '+%Y%m%d%H%M%S')
python src/train_model.py --timestamp "$timestamp"
```

Evaluate the selected model.

```
python src/evaluate_model.py --timestamp "$timestamp"
```

Move the outputs to their directories.

```
mv "${timestamp}_metrics.json" "metrics/${timestamp}_metrics.json"
mv "model_${timestamp}_dt_model.joblib" "models/model_${timestamp}_dt_model.joblib"
```

## What a Successful Run Looks Like

The sweep output will show all six configurations in order, with the best
flagged at each step, followed by a summary.

```
============================================================
Hyperparameter Sweep — 6 configurations
Train: 800 samples | Validation: 200 samples
============================================================
  Config 1: n_estimators= 50  learning_rate=0.10  | val F1: 0.9400 <-- best so far
  Config 2: n_estimators=100  learning_rate=0.10  | val F1: 0.9500 <-- best so far
  Config 3: n_estimators=200  learning_rate=0.10  | val F1: 0.9550 <-- best so far
  Config 4: n_estimators= 50  learning_rate=0.05  | val F1: 0.9200
  Config 5: n_estimators=100  learning_rate=0.05  | val F1: 0.9350
  Config 6: n_estimators=200  learning_rate=0.05  | val F1: 0.9450
============================================================
  Best config: n_estimators=200, learning_rate=0.10  (val F1: 0.9550)
============================================================

Final model saved: model_TIMESTAMP_dt_model.joblib
Config: n_estimators=200, learning_rate=0.10
```

On GitHub, after any push to main, the Actions tab will show the pipeline
running through all steps. The "Train Model" step log will contain the full
sweep table. The "Validate Model Quality" step will confirm the F1 score passes
the threshold. In the repository, the models/ folder will contain a new
timestamped joblib file and the metrics/ folder will contain the corresponding
JSON file, both committed automatically.

## Project Structure

```
IE7374-Github-Lab2/
    .github/
        workflows/
            model_retraining_on_push.yml
            model_calibration.yml
    src/
        __init__.py
        train_model.py
        evaluate_model.py
    models/
    metrics/
    data/
    .gitignore
    README.md
    requirements.txt
```

## CI/CD Pipeline Summary

model_retraining_on_push.yml triggers on every push to main. It generates a
timestamp, runs the hyperparameter sweep and saves the best model, evaluates it,
validates the F1 score against the 0.70 threshold, and if it passes, commits
and pushes the model and metrics files back to the repository.

model_calibration.yml runs the identical pipeline on a daily cron schedule at
00:00 UTC and also supports manual triggering via workflow_dispatch.
