# IE7374 Lab 2 — Automated ML Model Training and Versioning

## Overview

This lab implements an automated machine learning pipeline using GitHub Actions.
On every push to main, the pipeline trains a GradientBoostingClassifier, evaluates
it, runs a quality gate check against an F1 score threshold, and if the model passes,
commits the versioned model file and its metrics back into the repository automatically.
A second workflow runs the same pipeline on a daily schedule using cron.

## Changes Made to the Lab

Two categories of changes were made: changes to the model training code and
changes to the pipeline itself.

**Source code (src/train_model.py)**

The original lab used RandomForestClassifier trained on a generic synthetic
dataset with parameters that could generate zero samples. The model was changed
to GradientBoostingClassifier with n_estimators=100 and learning_rate=0.1.
The dataset name was changed to Drug Shortage Synthetic Dataset to reflect the
PharmTrack Sentinel project domain. n_samples was fixed from randint(0, 2000)
to randint(500, 2000) to prevent edge-case failures. n_features was increased
from 6 to 8 and n_informative from 3 to 5. Two additional parameters,
n_estimators and learning_rate, are now logged to MLflow so each run is fully
reproducible. The MLflow tracking URI was migrated from the deprecated file
store to SQLite to support current MLflow versions.

**Pipeline (model_retraining_on_push.yml and model_calibration.yml)**

A model quality gate step was added to both workflows between the evaluate
step and the commit step. After evaluation, the pipeline reads the F1 score
from the saved metrics JSON and checks it against a threshold of 0.70. If the
score is below the threshold, the pipeline exits with an error and skips the
commit entirely, preventing a poor model from being versioned into the
repository. If the score meets the threshold, the pipeline proceeds to commit
both the model file and its metrics with a timestamp-based filename.
This makes the automation conditional on quality, not just completion.

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

Install all dependencies. Key versions are scikit-learn 1.x, mlflow 2.x,
and joblib 1.x.

```
pip install -r requirements.txt
```

Create the output folders if they do not already exist.

```
mkdir -p models metrics data
```

## Running the Code Locally

The pipeline uses a timestamp to version every run. Generate one first, then
pass it to both scripts in the same session.

```
timestamp=$(date '+%Y%m%d%H%M%S')
echo "Timestamp: $timestamp"
```

Train the model.

```
python src/train_model.py --timestamp "$timestamp"
```

Evaluate the model.

```
python src/evaluate_model.py --timestamp "$timestamp"
```

Move the output files to their directories.

```
mv "${timestamp}_metrics.json" "metrics/${timestamp}_metrics.json"
mv "model_${timestamp}_dt_model.joblib" "models/model_${timestamp}_dt_model.joblib"
```

Manually check the quality gate by reading the metrics.

```
python -c "
import json
with open('metrics/${timestamp}_metrics.json') as f:
    data = json.load(f)
print('F1 Score:', data['F1_Score'])
"
```

## What a Successful Run Looks Like

After training you should see output similar to this.

```
Timestamp received: 20261007120000
Model saved: model_20261007120000_dt_model.joblib
```

After evaluation you should see this.

```
Metrics saved: 20261007120000_metrics.json
```

After checking the quality gate you should see an F1 score above 0.70,
for example F1 Score: 0.9812.

On GitHub, after any push to main, navigate to the Actions tab. The
"Model Retraining on Push to Main" workflow should show a green checkmark
on all steps including "Validate Model Quality". In the repository, the
models/ folder will contain a new timestamped joblib file and the metrics/
folder will contain the corresponding JSON file, both committed automatically
by the pipeline.

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
timestamp, trains the GradientBoostingClassifier, evaluates it, moves outputs
to their directories, validates F1 score against the 0.70 threshold, and if
the model passes, commits and pushes the model and metrics files back to the
repository using the GITHUB_TOKEN secret.

model_calibration.yml runs the identical pipeline on a daily cron schedule
at 00:00 UTC and also supports manual triggering via workflow_dispatch.
