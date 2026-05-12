#!/usr/bin/env python3
"""
Merged model training and evaluation pipeline for kinase datasets.

This script combines the four uploaded scripts without dropping their core logic:
  1. NB_KNN_RF_SVM_XGB_original_for_kinase.py
  2. NB_KNN_RF_SVM_XGB_smote_for_kinase.py
  3. NB_KNN_RF_SVM_XGB_under_for_kinase.py
  4. NB_KNN_RF_SVM_XGB_up_for_kinase.py

Preserved behavior:
  - Reads feature-selected CSV files under each feature-selection folder's
    selected_data_for_model subfolders.
  - Evaluates KNN, RF, NB, and XGB models. SVM remains optional because it was
    commented out in the original scripts.
  - Splits train/test by unique MappingID, using 30% of MappingID values for test.
  - Repeats random split 20 times with seeds 0..19 by default.
  - Uses ROC curve Youden index to choose the optimal threshold.
  - Reports AUC, Accuracy, Balanced Accuracy, and MCC mean/std.
  - Handles AUC < 0.5 using the same inversion logic as the original scripts.
  - Writes one result CSV per input CSV/model/resampling method.
  - Creates result folders such as knn_original, rf_smote, nb_under, xgb_up.
  - Deletes an existing result directory before writing, matching the original
    shutil.rmtree behavior, unless --no-clean-results is passed.

Notes about original differences preserved as configuration:
  - original script used fisher_exact_test in fs_dir.
  - smote/under/up scripts used fisher exact test in fs_dir.
  - up script contained a line forcing fs = 'fisher exact test' inside the loop.
    That exact legacy behavior is available through --legacy-up-only-fisher.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except Exception as exc:  # pragma: no cover - allows original mode without imblearn
    RandomOverSampler = None
    SMOTE = None
    RandomUnderSampler = None
    _IMBLEARN_IMPORT_ERROR = exc
else:
    _IMBLEARN_IMPORT_ERROR = None


# Defaults copied from the uploaded scripts.
DEFAULT_DIR_ORIGINAL = "D:\\2024\\20231218_kinase\\7_20240729_large_model_str_only\\1_data"
DEFAULT_DIR_SMOTE_UNDER = "D:\\2024\\20231218_kinase\\6_20240601_large kinase dataset\\3_model"
DEFAULT_DIR_UP = "D:\\2024\\20231218_kinase\\7_20240729_large_model_str_only\\1_data"

FS_DIRS_ORIGINAL = [
    "xgb_feature_selection",
    "rf_feature_selection",
    "auc_feature_selection",
    "fisher_exact_test",
]
FS_DIRS_SMOTE_UNDER_UP = [
    "xgb_feature_selection",
    "rf_feature_selection",
    "auc_feature_selection",
    "fisher exact test",
]

METHODS = ("original", "smote", "under", "up")
MODEL_ORDER = ("KNN", "RF", "NB", "XGB")


@dataclass(frozen=True)
class RunConfig:
    method: str
    input_dir: str
    fs_dirs: Sequence[str]
    models: Sequence[str]
    n_runs: int
    test_fraction: float
    mapping_id_col: str
    clean_results: bool
    include_svm: bool
    verbose: bool


def build_models(include_svm: bool = False) -> Dict[str, object]:
    """Create the model dictionary used by all original scripts."""
    models: Dict[str, object] = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RF": RandomForestClassifier(random_state=2024),
        "NB": GaussianNB(),
        "XGB": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            objective="binary:logistic",
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0,
            reg_lambda=1,
            n_estimators=100,
        ),
    }
    if include_svm:
        # SVM was present but commented out in the uploaded scripts.
        from sklearn.svm import SVC

        models["SVM"] = SVC(probability=True, random_state=2024)
    return models


def get_sampler_factory(method: str) -> Optional[Callable[[], object]]:
    """Return a new sampler factory for each run, or None for original mode."""
    if method == "original":
        return None
    if _IMBLEARN_IMPORT_ERROR is not None:
        raise ImportError(
            "imblearn is required for smote, under, and up methods. "
            "Install it with: pip install imbalanced-learn"
        ) from _IMBLEARN_IMPORT_ERROR
    if method == "smote":
        return lambda: SMOTE(random_state=2024)
    if method == "under":
        return lambda: RandomUnderSampler(random_state=2024)
    if method == "up":
        return lambda: RandomOverSampler(random_state=2024)
    raise ValueError(f"Unknown method: {method}")


def safe_std(values: Sequence[float]) -> float:
    """Match numpy std(ddof=1), but avoid noisy warnings for a single value."""
    if len(values) <= 1:
        return float("nan")
    return float(np.std(values, ddof=1))


def nonzero(values: Sequence[float]) -> List[float]:
    """Original scripts computed summaries after removing zeros."""
    return [float(v) for v in values if v != 0]


def split_by_mapping_id(
    data: pd.DataFrame,
    mapping_id_col: str,
    seed: int,
    test_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by unique MappingID exactly as in the uploaded scripts."""
    np.random.seed(seed)
    unique_ids = data[mapping_id_col].unique()
    test_size = int(len(unique_ids) * test_fraction)
    test_ids = np.random.choice(unique_ids, size=test_size, replace=False)
    testdata = data[data[mapping_id_col].isin(test_ids)]
    traindata = data[~data[mapping_id_col].isin(test_ids)]
    return traindata, testdata


def train_and_evaluate(
    file_path: str,
    model_template: object,
    method: str,
    n_runs: int = 20,
    test_fraction: float = 0.3,
    mapping_id_col: str = "MappingID",
    verbose: bool = True,
) -> Optional[Dict[str, float]]:
    """
    Train/evaluate one model on one CSV.

    This function merges the four train_and_evaluate functions. The only
    difference between the originals was the sampler:
      - original: no sampler
      - smote: SMOTE(random_state=2024)
      - under: RandomUnderSampler(random_state=2024)
      - up: RandomOverSampler(random_state=2024)
    """
    accuracies: List[float] = []
    aucs: List[float] = []
    balanced_accs: List[float] = []
    mccs: List[float] = []

    data = pd.read_csv(file_path)
    data.columns = data.columns[:-1].tolist() + ["Target"]
    data.iloc[:, 2:-1] = data.iloc[:, 2:-1].apply(pd.to_numeric, errors="coerce")

    if mapping_id_col not in data.columns:
        raise KeyError(
            f"Required split column '{mapping_id_col}' was not found in {file_path}. "
            f"Available columns include: {list(data.columns[:10])} ..."
        )

    sampler_factory = get_sampler_factory(method)

    for seed in range(n_runs):
        try:
            traindata, testdata = split_by_mapping_id(
                data=data,
                mapping_id_col=mapping_id_col,
                seed=seed,
                test_fraction=test_fraction,
            )

            # The uploaded scripts removed the first two columns after splitting
            # and dropped duplicate rows before training/testing.
            testdata = testdata.iloc[:, 2:].drop_duplicates()
            traindata = traindata.iloc[:, 2:].drop_duplicates()

            X_train = traindata.drop(columns=["Target"])
            y_train = traindata["Target"].astype("category").cat.codes
            X_test = testdata.drop(columns=["Target"])
            y_test = testdata["Target"].astype("category").cat.codes

            model = clone(model_template)
            if sampler_factory is not None:
                sampler = sampler_factory()
                X_train, y_train = sampler.fit_resample(X_train, y_train)

            model.fit(X_train, y_train)

            if hasattr(model, "predict_proba"):
                y_pred_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_prob = model.decision_function(X_test)

            # Compute ROC curve and apply optimal threshold using Youden index.
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            predict_results = (y_pred_prob > optimal_threshold).astype(int)

            auc_score = roc_auc_score(y_test, y_pred_prob)
            if verbose:
                print(auc_score)

            # Keep the original inversion logic for AUC < 0.5.
            if auc_score < 0.5:
                accuracies.append(1 - accuracy_score(y_test, predict_results))
                aucs.append(1 - auc_score)
                balanced_accs.append(1 - balanced_accuracy_score(y_test, predict_results))
                mccs.append(abs(matthews_corrcoef(y_test, predict_results)))
            else:
                accuracies.append(accuracy_score(y_test, predict_results))
                aucs.append(auc_score)
                balanced_accs.append(balanced_accuracy_score(y_test, predict_results))
                mccs.append(matthews_corrcoef(y_test, predict_results))

        except Exception as exc:
            print(f"Error during model training or evaluation: {exc}")
            continue

    if not accuracies:
        return None

    auc_values = nonzero(aucs)
    acc_values = nonzero(accuracies)
    bal_acc_values = nonzero(balanced_accs)
    mcc_values = nonzero(mccs)

    metrics_summary = {
        "AUC Mean": float(np.mean(auc_values)) if auc_values else float("nan"),
        "AUC Std": safe_std(auc_values),
        "Accuracy Mean": float(np.mean(acc_values)) if acc_values else float("nan"),
        "Accuracy Std": safe_std(acc_values),
        "Balanced Accuracy Mean": float(np.mean(bal_acc_values)) if bal_acc_values else float("nan"),
        "Balanced Accuracy Std": safe_std(bal_acc_values),
        "MCC Mean": float(np.mean(mcc_values)) if mcc_values else float("nan"),
        "MCC Std": safe_std(mcc_values),
    }
    return metrics_summary


def iter_target_dirs(base_input_dir: str, fs_dir: str) -> Iterable[str]:
    selected_data_dir = os.path.join(base_input_dir, fs_dir, "selected_data_for_model")
    if not os.path.isdir(selected_data_dir):
        print(f"Skipping missing directory: {selected_data_dir}")
        return
    for target in os.listdir(selected_data_dir):
        target_path = os.path.join(selected_data_dir, target)
        if os.path.isdir(target_path):
            yield target_path


def run_method(config: RunConfig) -> None:
    """Run one resampling method across all requested FS dirs, targets, and models."""
    start_time = time.time()
    all_models = build_models(include_svm=config.include_svm)
    selected_models = {name: all_models[name] for name in config.models}

    for fs_dir in config.fs_dirs:
        for target_path in iter_target_dirs(config.input_dir, fs_dir):
            for model_name, model_template in selected_models.items():
                results_dir = os.path.join(target_path, f"{model_name.lower()}_{config.method}")
                if os.path.exists(results_dir) and config.clean_results:
                    shutil.rmtree(results_dir, ignore_errors=True)
                os.makedirs(results_dir, exist_ok=True)

                for csv_file in glob.glob(os.path.join(target_path, "*.csv")):
                    print(csv_file)
                    metrics_summary = train_and_evaluate(
                        file_path=csv_file,
                        model_template=model_template,
                        method=config.method,
                        n_runs=config.n_runs,
                        test_fraction=config.test_fraction,
                        mapping_id_col=config.mapping_id_col,
                        verbose=config.verbose,
                    )
                    if metrics_summary:
                        metrics_df = pd.DataFrame(metrics_summary, index=[0])
                        metrics_file_name = (
                            f"{os.path.splitext(os.path.basename(csv_file))[0]}_"
                            f"{model_name}_{config.method}.csv"
                        )
                        metrics_df["data"] = metrics_file_name
                        metrics_df.to_csv(os.path.join(results_dir, metrics_file_name), index=False)

    elapsed_time = time.time() - start_time
    print(f"Total execution time for ALL_CODES_{config.method}: {elapsed_time:.2f} seconds")


def parse_list_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or None


def resolve_methods(raw_methods: Sequence[str]) -> List[str]:
    if "all" in raw_methods:
        return list(METHODS)
    methods = []
    for method in raw_methods:
        method = method.lower()
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Valid choices: {METHODS} or all")
        methods.append(method)
    return methods


def resolve_models(raw_models: Sequence[str], include_svm: bool) -> List[str]:
    available = list(MODEL_ORDER) + (["SVM"] if include_svm else [])
    if "all" in [m.lower() for m in raw_models]:
        return available
    models = []
    for model_name in raw_models:
        normalized = model_name.upper()
        if normalized not in available:
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
        models.append(normalized)
    return models


def default_input_dir_for_method(args: argparse.Namespace, method: str) -> str:
    shared_input_dir = args.input_dir
    if shared_input_dir:
        return shared_input_dir
    if method == "original":
        return args.input_dir_original or DEFAULT_DIR_ORIGINAL
    if method == "smote":
        return args.input_dir_smote or DEFAULT_DIR_SMOTE_UNDER
    if method == "under":
        return args.input_dir_under or DEFAULT_DIR_SMOTE_UNDER
    if method == "up":
        return args.input_dir_up or DEFAULT_DIR_UP
    raise ValueError(method)


def default_fs_dirs_for_method(args: argparse.Namespace, method: str) -> List[str]:
    override = parse_list_arg(args.fs_dirs)
    if override is not None:
        return override
    if method == "original":
        return list(FS_DIRS_ORIGINAL)
    if method in {"smote", "under", "up"}:
        fs_dirs = list(FS_DIRS_SMOTE_UNDER_UP)
        if method == "up" and args.legacy_up_only_fisher:
            # Original up script overwrote fs inside the loop:
            #     for fs in fs_dir:
            #         fs = 'fisher exact test'
            return ["fisher exact test"]
        return fs_dirs
    raise ValueError(method)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merged NB/KNN/RF/SVM/XGB kinase model evaluation pipeline."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to run: all, original, smote, under, up. Default: all.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to run: all, KNN, RF, NB, XGB. Add --include-svm to enable SVM.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="One dataset root directory for all methods. Overrides method-specific defaults.",
    )
    parser.add_argument("--input-dir-original", default=None, help="Dataset root for original mode.")
    parser.add_argument("--input-dir-smote", default=None, help="Dataset root for smote mode.")
    parser.add_argument("--input-dir-under", default=None, help="Dataset root for under mode.")
    parser.add_argument("--input-dir-up", default=None, help="Dataset root for up mode.")
    parser.add_argument(
        "--fs-dirs",
        default=None,
        help=(
            "Comma-separated feature-selection folders. If omitted, uses the original "
            "per-script defaults. Example: xgb_feature_selection,rf_feature_selection"
        ),
    )
    parser.add_argument("--n-runs", type=int, default=20, help="Number of random splits. Default: 20.")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="Fraction of unique MappingID values in test set. Default: 0.3.",
    )
    parser.add_argument(
        "--mapping-id-col",
        default="MappingID",
        help="Column used for grouped train/test split. Default: MappingID.",
    )
    parser.add_argument(
        "--include-svm",
        action="store_true",
        help="Enable SVM model. It was commented out in the original scripts.",
    )
    parser.add_argument(
        "--legacy-up-only-fisher",
        action="store_true",
        help="Replicate the original up script behavior that only runs fisher exact test.",
    )
    parser.add_argument(
        "--no-clean-results",
        action="store_true",
        help="Do not delete existing result directories before writing outputs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print each AUC value during model runs.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    methods = resolve_methods(args.methods)
    models = resolve_models(args.models, include_svm=args.include_svm)

    for method in methods:
        config = RunConfig(
            method=method,
            input_dir=default_input_dir_for_method(args, method),
            fs_dirs=default_fs_dirs_for_method(args, method),
            models=models,
            n_runs=args.n_runs,
            test_fraction=args.test_fraction,
            mapping_id_col=args.mapping_id_col,
            clean_results=not args.no_clean_results,
            include_svm=args.include_svm,
            verbose=not args.quiet,
        )
        print("=" * 80)
        print(f"Running method: {config.method}")
        print(f"Input dir: {config.input_dir}")
        print(f"FS dirs: {list(config.fs_dirs)}")
        print(f"Models: {list(config.models)}")
        print("=" * 80)
        run_method(config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
