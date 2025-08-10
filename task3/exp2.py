from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA


@dataclass
class Hparams:
    # DO NOT CHANGE THESE FOR FAIR COMPARISON
    val_split: float = 0.2
    seed: int = 42

    # Experiment settings, not hyperparameters
    num_rounds: int = 10000
    early_stopping_rounds: int = 500

    # Other hyperparameters
    pca_n_components = 1000

    # XGBoost hyperparameters
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.01
    xgb_num_parallel_tree: int = 1
    xgb_colsample_bynode: float = 0.5
    xgb_subsample: float = 0.5
    xgb_min_split_loss: float = 0.5
    xgb_min_child_weight: float = 2.0
    xgb_lambda = 2.0
    xgb_alpha = 0.0


def tfidf_to_np(df: pd.DataFrame):
    """Convert the tfidf CSVs to X array of features and y array of labels, ordered
    by id.
    """
    df = df.sort_index()

    if "label" in df.columns:
        y = df.pop("label").to_numpy()
    else:
        y = None

    X = df.to_numpy()
    return X, y


def fit_pca(HP: Hparams, train_X: np.ndarray, quiet: bool = False):
    """Fit PCA for dim reduction and report some stats."""
    model_pca = PCA(n_components=HP.pca_n_components, random_state=HP.seed)
    model_pca.fit(train_X)

    ratios = list(zip(range(HP.pca_n_components), model_pca.explained_variance_ratio_))
    ratios.sort(key=lambda x: x[1], reverse=True)

    if quiet:
        return model_pca

    print("Most Informative Dimensions:")
    for idx, val in ratios[:3]:
        print(f"  Dim {idx}: {val}")

    print("Noise:", model_pca.noise_variance_)
    print("Total Explained Variance:", sum(model_pca.explained_variance_))

    return model_pca


def train(
    HP: Hparams,
    *,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    quiet: bool = False,
):
    """Train model using hyperparameters."""
    model_pca = fit_pca(HP, train_X, quiet=quiet)

    # Transform the data
    t_train_X = model_pca.transform(train_X)
    t_val_X = model_pca.transform(val_X)
    # t_test_X = model_pca.transform(test_X)
    dtrain = xgb.DMatrix(t_train_X, label=train_y)
    dval = xgb.DMatrix(t_val_X, label=val_y)

    n_neg, n_pos = np.unique_counts(train_y).counts

    # See: https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
    # See: https://xgboost.readthedocs.io/en/stable/parameter.html
    xgb_params = {
        # Probably shouldn't adjust
        "validate_parameters": True,
        "tree_method": "hist",
        "device": "gpu",
        "eval_metric": "error",
        "sampling_method": "gradient_based",
        "objective": "binary:logistic",
        "random_state": HP.seed,
        #
        # Hyperparameters
        "max_depth": HP.xgb_max_depth,
        "learning_rate": HP.xgb_learning_rate,
        "num_parallel_tree": HP.xgb_num_parallel_tree,
        "colsample_bynode": HP.xgb_colsample_bynode,
        "subsample": HP.xgb_subsample,
        "min_split_loss": HP.xgb_min_split_loss,
        "min_child_weight": HP.xgb_min_child_weight,
        "lambda": HP.xgb_lambda,
        "alpha": HP.xgb_alpha,
        #
        # Computed based on data
        # sum(negative instances) / sum(positive instances)
        "scale_pos_weight": n_neg / n_pos,
    }

    # Last set is used by xgb's early stopping.
    eval_list = [(dtrain, "train"), (dval, "val")]

    results = {}
    model_xgb = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=HP.num_rounds,
        evals=eval_list,
        evals_result=results,
        early_stopping_rounds=HP.early_stopping_rounds,
        # verbose_eval=200,
        verbose_eval=False,
    )

    best_iter = model_xgb.best_iteration
    val_err = results["val"]["error"][best_iter]

    if not quiet:
        print(f"Best iteration: {best_iter}")
        print(f"Validation error at best iteration: {val_err}")

    return dict(
        val_err=val_err,
        model_xgb=model_xgb,
        model_pca=model_pca,
    )


def inference(
    test_X: np.ndarray,
    *,
    model_xgb: xgb.Booster,
    model_pca: PCA,
):
    """Run inference on the test set."""
    t_test_X = model_pca.transform(test_X)
    dtest = xgb.DMatrix(t_test_X)

    pred_y = model_xgb.predict(dtest, iteration_range=(0, model_xgb.best_iteration + 1))
    return pred_y
