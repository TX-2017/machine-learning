# -*- coding: utf-8 -*-
"""
Merged feature-selection pipeline
"""

import argparse
import os
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================
# 0. User settings / original paths
# =============================

# Original AUC script path:
DEFAULT_AUC_INPUT_DIR = r"C:\Users\xut2\Desktop\demo"

# Original Fisher/t-test script path:
DEFAULT_FISHER_INPUT_DIR = r"D:\2024\20231218_kinase\7_20240729_large_model_str_only\1_data"

# Original RF/XGB scripts path:
DEFAULT_MODEL_INPUT_DIR = r"D:\2024\20231218_kinase\6_20240601_large kinase dataset\3_model"

# Original thresholds preserved.
AUC_STAT_THRESHOLD = 0.55
AUC_SELECTED_THRESHOLDS = [round(x, 2) for x in np.arange(0.51, 0.61, 0.01)]
FISHER_STAT_THRESHOLD = 0.05
FISHER_SELECTED_THRESHOLDS = [round(x, 2) for x in np.arange(0.01, 0.06, 0.01)]
NUM_FEATURES_LIST = list(range(100, 1001, 100))

# Original XGBoost parameter set preserved.
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "error",
    "max_depth": 3,
    "eta": 0.01,
    "gamma": 1,
    "colsample_bytree": 0.5,
    "min_child_weight": 1,
}


# =============================
# 1. Shared helpers
# =============================

def today_str() -> str:
    """Return YYYY-MM-DD string, matching the original output naming style."""
    return str(date.today())


def list_csv_files(input_dir: str) -> List[str]:
    """Return CSV file names in an input directory."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")
    return csv_files


def ensure_dir(path: str) -> str:
    """Create directory if needed and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def print_basic_overview(file_name: str, df: pd.DataFrame) -> None:
    """Print the same style of diagnostic information used in the original scripts."""
    print("=" * 80)
    print(f"File: {file_name}")
    print(f"First 10 columns: {df.columns[:10].tolist()}")
    print(f"Last column: {df.columns[-1]}")
    print(f"Shape: {df.shape}")
    if "Entrez_ID" in df.columns:
        print(f"Unique Entrez_ID: {df['Entrez_ID'].nunique()}")
    if "MappingID" in df.columns:
        print(f"Unique MappingID: {df['MappingID'].nunique()}")


def load_csvs_for_stat_methods(input_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load CSV files for AUC and Fisher/t-test workflows.

    This preserves the original behavior:
    - read each CSV with header
    - convert df.iloc[:, 2:] to float
    - rename the last column to the file name
    """
    csv_files = list_csv_files(input_dir)
    print(f"CSV files found in {input_dir}: {csv_files}")

    object_file_list: Dict[str, pd.DataFrame] = {}
    for file_name in csv_files:
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path)
        print_basic_overview(file_name, df)
        df.iloc[:, 2:] = df.iloc[:, 2:].astype(float)
        df.rename(columns={df.columns[-1]: file_name}, inplace=True)
        print(f"After conversion and target rename: last column = {df.columns[-1]}, shape = {df.shape}")
        object_file_list[file_name] = df

    last_file = csv_files[-1]
    last_df = object_file_list[last_file]
    print("=" * 80)
    print(f"Shape of last dataframe: {last_df.shape}")
    print(f"First 10 columns of last dataframe: {last_df.columns[:10].tolist()}")
    print(f"Last columns of last dataframe: {last_df.columns[-min(48, last_df.shape[1]):].tolist()}")
    return object_file_list


def load_csvs_for_model_methods(input_dir: str) -> Tuple[List[str], List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Load CSV files for RF and XGB workflows.

    This preserves the original behavior:
    - keep one converted dataframe list for modeling
    - keep one original dataframe list for selected_data export
    - rename the last column to the CSV file name in both copies
    - convert df.iloc[:, 2:] to numeric in the modeling copy
    """
    csv_files = list_csv_files(input_dir)
    print(f"CSV files found in {input_dir}: {csv_files}")

    data_frames: List[pd.DataFrame] = []
    data_frames_ori: List[pd.DataFrame] = []

    for file_name in csv_files:
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path, header=0)
        df_original = pd.read_csv(file_path, header=0)

        df.columns = list(df.columns[:-1]) + [file_name]
        df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric)

        df_original.columns = list(df_original.columns[:-1]) + [file_name]

        data_frames.append(df)
        data_frames_ori.append(df_original)

    if data_frames:
        print("=" * 80)
        print("Data shape:", data_frames[-1].shape)
        print("First 10 columns:", data_frames[-1].columns[:10].tolist())
        print("Last 10 columns:", data_frames[-1].columns[-10:].tolist())

    return csv_files, data_frames, data_frames_ori


def select_and_save_model_data(
    original_df: pd.DataFrame,
    selected_features: Sequence[str],
    output_path: str,
) -> None:
    """Save first two columns + selected features + last target column."""
    selected_columns = [original_df.columns[0], original_df.columns[1]] + list(selected_features) + [original_df.columns[-1]]
    output_data = original_df.loc[:, selected_columns]
    output_data.to_csv(output_path, index=False)


# =============================
# 2. AUC feature selection
# =============================

def getROC_AUC(probs: np.ndarray, true_Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Original custom AUC function preserved from AUC_ROC_FS.py."""
    sorted_indices = np.argsort(probs)[::-1]
    sorted_y = true_Y[sorted_indices]

    stack_x = np.cumsum(sorted_y == 1) / np.sum(sorted_y == 1)
    stack_y = np.cumsum(sorted_y == 0) / np.sum(sorted_y == 0)

    auc = np.sum((stack_x[1:] - stack_x[:-1]) * stack_y[1:])
    return stack_x, stack_y, auc


def run_auc_feature_selection(input_dir: str) -> Dict[str, pd.DataFrame]:
    """Run the merged AUC ROC feature-selection workflow."""
    from sklearn.metrics import roc_auc_score

    object_file_list = load_csvs_for_stat_methods(input_dir)
    output_file = ensure_dir(os.path.join(input_dir, "auc_feature_selection"))

    object_file_list_select: Dict[str, pd.DataFrame] = {}
    for file_name, df in object_file_list.items():
        true_y = df.iloc[:, -1].values
        auc_all_1: List[float] = []
        auc_all_2: List[float] = []

        for j in range(2, df.shape[1] - 1):
            probs = df.iloc[:, j].values
            _, _, auc_1 = getROC_AUC(probs, true_y)
            if auc_1 < 0.5:
                auc_1 = 1 - auc_1

            auc_2 = roc_auc_score(true_y, probs)
            if auc_2 < 0.5:
                auc_2 = 1 - auc_2

            auc_all_1.append(auc_1)
            auc_all_2.append(auc_2)

        all_auc_merge = pd.DataFrame(
            {
                "auc_without_proc": auc_all_1,
                "auc_with_proc": auc_all_2,
                file_name: df.columns[2:-1],
            }
        )

        ranked_auc = all_auc_merge.sort_values("auc_with_proc", ascending=False)
        object_file_list_select[file_name] = ranked_auc
        ranked_auc.to_csv(os.path.join(output_file, f"{today_str()}-{file_name}_AUC.csv"), index=True)

    # AUC >= 0.55 feature statistics.
    stat_dir = ensure_dir(os.path.join(output_file, "AUC_features_stat"))
    for file_name, df in object_file_list_select.items():
        selected = df[df["auc_with_proc"] >= AUC_STAT_THRESHOLD]
        out_file = os.path.join(stat_dir, f"{today_str()}-{file_name}-{AUC_STAT_THRESHOLD:.2f}.csv")
        selected.to_csv(out_file, index=False)

    # Selected model-ready data for AUC thresholds.
    dir_path_target = ensure_dir(os.path.join(output_file, "selected_data_for_model"))
    for threshold in AUC_SELECTED_THRESHOLDS:
        ensure_dir(os.path.join(dir_path_target, f"pvalue_{threshold:.2f}"))

    last_category_assay: Optional[List[str]] = None
    for file_name, ranked_df in object_file_list_select.items():
        source_df = object_file_list[file_name]
        for threshold in AUC_SELECTED_THRESHOLDS:
            category_assay = ranked_df[ranked_df["auc_with_proc"] >= threshold][file_name].tolist()
            last_category_assay = category_assay

            output_data = source_df.iloc[:, :2].copy()
            output_data = pd.concat([output_data, source_df[category_assay]], axis=1)
            output_data = pd.concat([output_data, source_df.iloc[:, -1]], axis=1)
            output_data.rename(columns={output_data.columns[-1]: file_name}, inplace=True)

            if not output_data.empty:
                output_path = os.path.join(
                    dir_path_target,
                    f"pvalue_{threshold:.2f}",
                    f"{today_str()}-{file_name}-{threshold:.2f}.csv",
                )
                output_data.to_csv(output_path, index=False)

    if last_category_assay:
        print("AUC selected feature preview:", last_category_assay[:5])
    else:
        print("AUC selected feature preview: No categories selected")

    return object_file_list_select


# =============================
# 3. Fisher exact test + t-test feature selection
# =============================

def run_fisher_ttest_feature_selection(input_dir: str) -> Dict[str, pd.DataFrame]:
    """Run Fisher exact test for V* columns and t-test for non-V columns."""
    from scipy import stats

    object_file_list = load_csvs_for_stat_methods(input_dir)
    output_file = ensure_dir(os.path.join(input_dir, "fisher exact test"))

    object_file_list_select: Dict[str, pd.DataFrame] = {}
    last_f: Optional[pd.DataFrame] = None

    for file_name, df in object_file_list.items():
        a1 = df.copy()
        v_cols = [col for col in a1.columns if str(col).startswith("V")]
        a = a1[v_cols + [a1.columns[-1]]]
        print("Fisher columns:", a.columns.tolist())

        fisher_pvalues: List[float] = []
        for mm in range(a.shape[1] - 1):
            tox_target = a[(a.iloc[:, mm] == 1) & (a.iloc[:, -1] == 1)]
            tox_non_target = a[(a.iloc[:, mm] == 0) & (a.iloc[:, -1] == 1)]
            non_tox_target = a[(a.iloc[:, mm] == 1) & (a.iloc[:, -1] == 0)]
            non_tox_non_target = a[(a.iloc[:, mm] == 0) & (a.iloc[:, -1] == 0)]

            contingency_table = [
                [len(tox_target), len(non_tox_target)],
                [len(tox_non_target), len(non_tox_non_target)],
            ]
            _, p_value = stats.fisher_exact(contingency_table)
            fisher_pvalues.append(p_value)

        fisher_df = pd.DataFrame({"pvalue": fisher_pvalues, file_name: a.columns[:-1]})

        # T-test for non-V columns after the first two columns, preserving original logic.
        avirus = a1[[col for col in a1.columns if not str(col).startswith("V")]]
        avirus = avirus.iloc[:, 2:]

        ttest_pvalues: List[float] = []
        for col in avirus.columns[:-1]:
            _, p_value = stats.ttest_ind(
                avirus[avirus.iloc[:, -1] == 1][col],
                avirus[avirus.iloc[:, -1] == 0][col],
            )
            ttest_pvalues.append(p_value)

        ttest_df = pd.DataFrame({"pvalue": ttest_pvalues, file_name: avirus.columns[:-1]})

        merged_pvalues = pd.concat([fisher_df, ttest_df])
        ranked_pvalues = merged_pvalues.sort_values("pvalue")

        object_file_list_select[file_name] = ranked_pvalues
        last_f = ranked_pvalues
        ranked_pvalues.to_csv(os.path.join(output_file, f"{today_str()}-{file_name}"), index=False)

    if last_f is not None and not last_f.empty:
        print(f"Max p-value: {last_f['pvalue'].max()}")
        print(f"Min p-value: {last_f['pvalue'].min()}")
        print(f"Head of ranked p-values:\n{last_f.head()}")

    # p <= 0.05 feature statistics.
    stat_dir = ensure_dir(os.path.join(output_file, "pvalue_features_stat"))
    last_selected: Optional[pd.DataFrame] = None
    for file_name, df in object_file_list_select.items():
        selected = df[df["pvalue"] <= FISHER_STAT_THRESHOLD]
        last_selected = selected
        out_file = os.path.join(stat_dir, f"{today_str()}-{file_name}-{FISHER_STAT_THRESHOLD:.2f}.csv")
        selected.to_csv(out_file, index=False)

    if last_selected is not None:
        print(f"Shape of p <= {FISHER_STAT_THRESHOLD:.2f}: {last_selected.shape}")
        print(f"Head of p <= {FISHER_STAT_THRESHOLD:.2f}:\n{last_selected.head()}")

    # Selected model-ready data for p-value thresholds.
    dir_path_target = ensure_dir(os.path.join(output_file, "selected_data_for_model"))
    for threshold in FISHER_SELECTED_THRESHOLDS:
        ensure_dir(os.path.join(dir_path_target, f"pvalue_{threshold:.2f}"))

    for file_name, df in object_file_list.items():
        for threshold in FISHER_SELECTED_THRESHOLDS:
            category_assay = object_file_list_select[file_name][
                object_file_list_select[file_name]["pvalue"] <= threshold
            ][file_name].tolist()
            output_data = df[[df.columns[0], df.columns[1]] + category_assay + [df.columns[-1]]]
            output_path = os.path.join(
                dir_path_target,
                f"pvalue_{threshold:.2f}",
                f"{today_str()}-{df.columns[-1]}-{threshold:.2f}.csv",
            )
            output_data.to_csv(output_path, index=False)

    if object_file_list:
        last_df = next(reversed(object_file_list.values()))
        print(f"First two columns: {last_df.columns[:2].tolist()}")

    return object_file_list_select


# =============================
# 4. Random Forest feature selection
# =============================

def run_rf_feature_selection(input_dir: str) -> List[pd.DataFrame]:
    """Run RandomForestClassifier feature selection and export selected data."""
    from sklearn.ensemble import RandomForestClassifier

    csv_files, data_frames, data_frames_ori = load_csvs_for_model_methods(input_dir)
    output_root = ensure_dir(os.path.join(input_dir, "rf_feature_selection"))

    selected_features_list: List[pd.DataFrame] = []
    for df_index, df in enumerate(data_frames):
        data = df.copy()
        data.columns = list(data.columns[:-1]) + ["target"]
        data["target"] = pd.to_numeric(data["target"])

        x_data, y_data = data.iloc[:, 2:-1], data.iloc[:, -1]
        model = RandomForestClassifier()
        model.fit(x_data, y_data)

        importances = model.feature_importances_
        feat_importances = pd.DataFrame({"Feature": x_data.columns, "Importance": importances})
        feat_importances = feat_importances.sort_values("Importance", ascending=False)
        selected_features_list.append(feat_importances)

        feat_importances.to_csv(os.path.join(output_root, csv_files[df_index]), index=False)

    selected_data_dir = ensure_dir(os.path.join(output_root, "selected_data"))
    for num_features in NUM_FEATURES_LIST:
        ensure_dir(os.path.join(selected_data_dir, f"rf_{num_features}"))

    for file_index, df in enumerate(data_frames_ori):
        for num_features in NUM_FEATURES_LIST:
            top_features = selected_features_list[file_index]["Feature"].head(num_features)
            output_file_name = f"{today_str()}-{df.columns[-1]}-{num_features}.csv"
            output_path = os.path.join(selected_data_dir, f"rf_{num_features}", output_file_name)
            select_and_save_model_data(df, top_features, output_path)

    return selected_features_list


# =============================
# 5. XGBoost feature selection
# =============================

def run_xgb_feature_selection(input_dir: str) -> List[pd.DataFrame]:
    """Run XGBClassifier feature selection and export selected data."""
    from xgboost import XGBClassifier

    csv_files, data_frames, data_frames_ori = load_csvs_for_model_methods(input_dir)
    output_root = ensure_dir(os.path.join(input_dir, "xgb_feature_selection"))

    selected_features_list: List[pd.DataFrame] = []
    for df_index, df in enumerate(data_frames):
        data = df.copy()
        data.columns = list(data.columns[:-1]) + ["target"]
        data["target"] = pd.to_numeric(data["target"])

        x_data, y_data = data.iloc[:, 2:-1], data.iloc[:, -1]
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(x_data, y_data)

        importances = model.feature_importances_
        feat_importances = pd.DataFrame({"Feature": x_data.columns, "Gain": importances})
        feat_importances = feat_importances.sort_values("Gain", ascending=False)
        selected_features_list.append(feat_importances)

        feat_importances.to_csv(os.path.join(output_root, csv_files[df_index]), index=False)

    selected_data_dir = ensure_dir(os.path.join(output_root, "selected_data"))
    for num_features in NUM_FEATURES_LIST:
        ensure_dir(os.path.join(selected_data_dir, f"xgb_{num_features}"))

    for file_index, df in enumerate(data_frames_ori):
        for num_features in NUM_FEATURES_LIST:
            top_features = selected_features_list[file_index]["Feature"].head(num_features)
            output_file_name = f"{today_str()}-{df.columns[-1]}-{num_features}.csv"
            output_path = os.path.join(selected_data_dir, f"xgb_{num_features}", output_file_name)
            select_and_save_model_data(df, top_features, output_path)

    return selected_features_list


# =============================
# 6. Command-line interface
# =============================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merged feature selection pipeline: AUC, Fisher/t-test, RF, and XGB."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=["all", "auc", "fisher", "rf", "xgb"],
        help="Which workflow(s) to run. Default: all.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Use one directory for all selected methods. Overrides method-specific default dirs.",
    )
    parser.add_argument(
        "--auc-dir",
        default=DEFAULT_AUC_INPUT_DIR,
        help="Input directory for AUC workflow.",
    )
    parser.add_argument(
        "--fisher-dir",
        default=DEFAULT_FISHER_INPUT_DIR,
        help="Input directory for Fisher/t-test workflow.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_INPUT_DIR,
        help="Input directory for RF and XGB workflows.",
    )
    return parser.parse_args()


def normalize_methods(methods: Iterable[str]) -> List[str]:
    methods = list(methods)
    if "all" in methods:
        return ["auc", "fisher", "rf", "xgb"]
    # Preserve user order and remove duplicates.
    normalized: List[str] = []
    for method in methods:
        if method not in normalized:
            normalized.append(method)
    return normalized


def main() -> None:
    args = parse_args()
    methods = normalize_methods(args.methods)

    auc_dir = args.input_dir or args.auc_dir
    fisher_dir = args.input_dir or args.fisher_dir
    model_dir = args.input_dir or args.model_dir

    print("Selected methods:", methods)
    print("AUC input dir:", auc_dir)
    print("Fisher/t-test input dir:", fisher_dir)
    print("RF/XGB input dir:", model_dir)

    if "auc" in methods:
        print("\nRunning AUC feature selection...")
        run_auc_feature_selection(auc_dir)

    if "fisher" in methods:
        print("\nRunning Fisher exact test + t-test feature selection...")
        run_fisher_ttest_feature_selection(fisher_dir)

    if "rf" in methods:
        print("\nRunning Random Forest feature selection...")
        run_rf_feature_selection(model_dir)

    if "xgb" in methods:
        print("\nRunning XGBoost feature selection...")
        run_xgb_feature_selection(model_dir)

    print("\nDone. All selected workflows finished.")


if __name__ == "__main__":
    main()
