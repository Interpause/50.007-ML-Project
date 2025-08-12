import json
import os
import pickle
import re
from datetime import datetime
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def clean_text(text):
    """Comprehensive text cleaning function for social media/web content"""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs (improved pattern)
    text = re.sub(r"https?://\S+|www\.\S+|http\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove mentions and hashtags (keep the text but remove @ and #)
    text = re.sub(r"[@#](\w+)", r"\1", text)

    # Replace contractions with expanded forms
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
        "it's": "it is",
        "that's": "that is",
        "what's": "what is",
        "where's": "where is",
        "how's": "how is",
        "there's": "there is",
        "here's": "here is",
    }
    for contraction, expansion in contractions.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)

    # Remove numbers but keep words with numbers (e.g., keep "covid19" but remove "123")
    text = re.sub(r"\b\d+\b", "", text)

    # Remove special characters and punctuation (but keep letters and spaces)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove single characters (except 'a' and 'i')
    text = re.sub(r"\b[b-hj-z]\b", "", text, flags=re.IGNORECASE)

    # Remove extra whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_clean_data() -> Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """Load and clean the data once to avoid repeating expensive text cleaning."""
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Identify text column
    text_col = [c for c in train_df.columns if c not in ("id", "label")][0]
    print(f"Using text column: {text_col}")

    # Clean text once (this is expensive, so we do it only once)
    print("Cleaning text...")
    X_train_text = train_df[text_col].apply(clean_text)
    X_test_text = test_df[text_col].apply(clean_text)
    y_train = train_df["label"].values
    test_ids = test_df["id"].values

    print(f"Training samples: {len(X_train_text)}")
    print(f"Test samples: {len(X_test_text)}")

    return X_train_text, X_test_text, y_train, test_ids


def create_pipeline(trial: optuna.Trial) -> Pipeline:
    """Create a pipeline with hyperparameters suggested by Optuna."""

    # TfidfVectorizer hyperparameters
    max_features = trial.suggest_categorical(
        "max_features", [10000, 15000, 20000, 30000, 50000]
    )
    min_df = trial.suggest_categorical("min_df", [1, 2, 3, 5])
    max_df = trial.suggest_float("max_df", 0.8, 0.99)
    ngram_max = trial.suggest_categorical("ngram_max", [2, 3, 4])
    sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False])

    # SelectKBest hyperparameters
    k_features = trial.suggest_categorical(
        "k_features", [3000, 5000, 8000, 10000, 15000]
    )

    # LogisticRegression hyperparameters
    C = trial.suggest_float("C", 0.01, 100.0, log=True)
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga"])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    max_iter = trial.suggest_categorical("max_iter", [1000, 2000, 5000])

    # Handle penalty and solver compatibility
    if penalty == "elasticnet" and solver != "saga":
        solver = "saga"
    elif penalty == "l1" and solver == "lbfgs":
        solver = "liblinear"

    # Add l1_ratio for elasticnet
    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)

    # Create pipeline
    pipeline_steps = [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, ngram_max),
                min_df=min_df,
                max_df=max_df,
                stop_words="english",
                sublinear_tf=sublinear_tf,
                strip_accents="unicode",
            ),
        ),
        ("feature_selection", SelectKBest(score_func=chi2, k=k_features)),
        (
            "classifier",
            LogisticRegression(
                C=C,
                solver=solver,
                penalty=penalty,
                max_iter=max_iter,
                random_state=42,
                class_weight="balanced",
                l1_ratio=l1_ratio if penalty == "elasticnet" else None,
            ),
        ),
    ]

    return Pipeline(pipeline_steps)


def objective(
    trial: optuna.Trial, X_train_text: pd.Series, y_train: np.ndarray
) -> float:
    """Objective function for Optuna optimization."""

    try:
        # Create pipeline with suggested hyperparameters
        pipeline = create_pipeline(trial)

        # Use stratified cross-validation for robust evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Use macro F1 score as the metric
        f1_macro_scorer = make_scorer(f1_score, average="macro")

        # Perform cross-validation
        cv_scores = cross_val_score(
            pipeline, X_train_text, y_train, cv=cv, scoring=f1_macro_scorer, n_jobs=-1
        )

        # Return mean CV score
        mean_score = np.mean(cv_scores)

        # Log progress every 10 trials
        if trial.number % 10 == 0:
            print(
                f"Trial {trial.number}: CV F1 = {mean_score:.4f} ± {np.std(cv_scores):.4f}"
            )

        return mean_score

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def run_optuna_optimization(
    X_train_text: pd.Series,
    y_train: np.ndarray,
    n_trials: int = 100,
    timeout: int = 3600,
) -> optuna.Study:
    """Run Optuna hyperparameter optimization."""

    print(
        f"Starting Optuna optimization with {n_trials} trials or {timeout / 60:.0f} minutes timeout..."
    )

    # Create study
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train_text, y_train),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    print(f"\nOptimization completed!")
    print(f"Best CV F1 score: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study


def train_final_model(
    study: optuna.Study, X_train_text: pd.Series, y_train: np.ndarray
) -> Pipeline:
    """Train final model with best hyperparameters on full training data."""

    print("\nTraining final model with best hyperparameters...")

    # Create pipeline with best parameters
    best_trial = study.best_trial
    best_pipeline = create_pipeline(best_trial)

    # Train on full training data
    best_pipeline.fit(X_train_text, y_train)

    return best_pipeline


def evaluate_model(
    pipeline: Pipeline, X_train_text: pd.Series, y_train: np.ndarray
) -> dict:
    """Evaluate the final model using cross-validation."""

    print("Evaluating final model...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_macro_scorer = make_scorer(f1_score, average="macro")

    cv_scores = cross_val_score(
        pipeline, X_train_text, y_train, cv=cv, scoring=f1_macro_scorer, n_jobs=-1
    )

    results = {
        "cv_scores": cv_scores.tolist(),
        "mean_cv_f1": float(np.mean(cv_scores)),
        "std_cv_f1": float(np.std(cv_scores)),
        "cv_range": (float(np.min(cv_scores)), float(np.max(cv_scores))),
    }

    print(
        f"Final model CV F1: {results['mean_cv_f1']:.4f} ± {results['std_cv_f1']:.4f}"
    )
    print(f"CV range: [{results['cv_range'][0]:.4f}, {results['cv_range'][1]:.4f}]")

    return results


def make_predictions(
    pipeline: Pipeline, X_test_text: pd.Series, test_ids: np.ndarray
) -> pd.DataFrame:
    """Make predictions on test set."""

    print("Making predictions on test set...")

    # Get prediction probabilities
    test_probabilities = pipeline.predict_proba(X_test_text)[:, 1]

    # Convert to binary predictions
    test_predictions = (test_probabilities > 0.5).astype(int)

    # Create submission dataframe
    submission_df = pd.DataFrame({"id": test_ids, "label": test_predictions})

    print(
        f"Test predictions - Positive class: {np.sum(test_predictions)}/{len(test_predictions)} ({np.mean(test_predictions) * 100:.1f}%)"
    )

    return submission_df


def save_results(
    study: optuna.Study,
    pipeline: Pipeline,
    evaluation_results: dict,
    submission_df: pd.DataFrame,
    experiment_name: str,
):
    """Save all results and artifacts."""

    # Create experiment directory
    log_dir = f"./logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Save Optuna study
    study_file = os.path.join(log_dir, "optuna_study.pkl")
    with open(study_file, "wb") as f:
        pickle.dump(study, f)

    # Save best pipeline
    pipeline_file = os.path.join(log_dir, "best_pipeline.pkl")
    with open(pipeline_file, "wb") as f:
        pickle.dump(pipeline, f)

    # Save experiment metadata
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "best_cv_f1": study.best_value,
        "best_params": study.best_params,
        "evaluation_results": evaluation_results,
        "total_trials": len(study.trials),
        "n_complete": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "n_failed": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        ),
    }

    metadata_file = os.path.join(log_dir, "experiment_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save submission
    submission_file = f"{datetime.now().strftime('%Y%m%d-%H%M')}-logistic-optuna.csv"
    submission_df.to_csv(submission_file, index=False)

    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_file = os.path.join(log_dir, "trials.csv")
    trials_df.to_csv(trials_file, index=False)

    print(f"\nResults saved:")
    print(f"  Experiment directory: {log_dir}")
    print(f"  Submission file: {submission_file}")
    print(f"  Best pipeline: {pipeline_file}")
    print(f"  Optuna study: {study_file}")


def main():
    """Main execution function."""

    # Setup experiment
    experiment_name = f"logistic_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Starting experiment: {experiment_name}")

    # Load and clean data (expensive operation done once)
    X_train_text, X_test_text, y_train, test_ids = load_and_clean_data()

    # Run hyperparameter optimization
    study = run_optuna_optimization(X_train_text, y_train, n_trials=200, timeout=3600)

    # Train final model
    best_pipeline = train_final_model(study, X_train_text, y_train)

    # Evaluate model
    evaluation_results = evaluate_model(best_pipeline, X_train_text, y_train)

    # Make predictions
    submission_df = make_predictions(best_pipeline, X_test_text, test_ids)

    # Save all results
    save_results(
        study, best_pipeline, evaluation_results, submission_df, experiment_name
    )

    print(f"\nExperiment {experiment_name} completed successfully!")

    return study, best_pipeline, evaluation_results, submission_df


if __name__ == "__main__":
    study, pipeline, results, submission = main()
