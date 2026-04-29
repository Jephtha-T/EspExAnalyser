# Train and run extraction-level classifiers.

import argparse
import csv
import json
import math
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Data_Export import build_event_row, build_frame_rows, build_summary_row
from Frame_Extraction import DEFAULT_EXTRACTION_FPS, safe_fps, shot_duration_seconds

base_dir = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.join(base_dir, "Analysis")

default_summary_csv = os.path.join(analysis_dir, "training_data.csv")
default_events_csv = os.path.join(analysis_dir, "training_data_events.csv")
default_model_path = os.path.join(analysis_dir, "extraction_model.joblib")
default_rf_model_path = os.path.join(analysis_dir, "extraction_model_rf.joblib")
default_svm_model_path = os.path.join(analysis_dir, "extraction_model_svm.joblib")
default_logistic_model_path = os.path.join(analysis_dir, "extraction_model_logistic.joblib")

legacy_feature_columns = [
    "shot_time",
    "blonding_rate",
    "channeling_range",
    "channeling_range_norm",
    "channeling_coverage_norm",
]

identifier_columns = {"video_id", "label"}

class_name_map = {
    0: "under",
    1: "ideal",
    2: "over",
}

feature_set_default = "auto"
model_type_default = "all"

summary_compact_feature_columns = (
    "summary:shot_time",
    "summary:blonding_rate_norm",
    "summary:channeling_range_norm",
    "summary:channeling_coverage_norm",
    "summary:channel_quality_score",
    "summary:diag_score_uneven_extraction_long_high_channeling",
    "summary:diag_flag_count",
    "summary:diag_flag_likely_over_extraction_long_slow_blonding",
    "summary:diag_flag_uneven_extraction_long_high_channeling",
)

summary_compact_best_feature_columns = (
    "summary:channel_quality_score",
    "summary:diag_score_uneven_extraction_long_high_channeling",
    "summary:diag_flag_count",
    "summary:diag_flag_likely_over_extraction_long_slow_blonding",
    "summary:diag_flag_uneven_extraction_long_high_channeling",
)

supported_feature_sets = ("auto", "summary", "summary_compact", "summary_compact_best", "events", "combined")
supported_model_types = ("all", "both", "svm", "random_forest", "logistic")
default_test_size = 0.2
default_random_state = 42


def _safe_float(value, default=0.0):
    try:
        number = float(value)
        if math.isnan(number):
            return float(default)
        return number
    except (TypeError, ValueError):
        return float(default)


def _safe_float_or_nan(value):
    try:
        number = float(value)
        if math.isnan(number):
            return np.nan
        return number
    except (TypeError, ValueError):
        return np.nan


def _safe_int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _ideal_binary_labels(values):
    array = np.array(values, dtype=np.int64)
    return (array == 1).astype(np.int64)


def _ideal_vs_nonideal_accuracy(y_true, y_pred):
    return accuracy_score(_ideal_binary_labels(y_true), _ideal_binary_labels(y_pred))


def _ideal_vs_nonideal_balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(_ideal_binary_labels(y_true), _ideal_binary_labels(y_pred))


def _ideal_vs_nonideal_macro_f1(y_true, y_pred):
    return f1_score(
        _ideal_binary_labels(y_true),
        _ideal_binary_labels(y_pred),
        average="macro",
        zero_division=0,
    )


evaluation_scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro",
    "ideal_vs_nonideal_accuracy": make_scorer(_ideal_vs_nonideal_accuracy),
    "ideal_vs_nonideal_balanced_accuracy": make_scorer(_ideal_vs_nonideal_balanced_accuracy),
    "ideal_vs_nonideal_macro_f1": make_scorer(_ideal_vs_nonideal_macro_f1),
}


def _compute_channeling_metrics(channeling_stats):
    channeling_stats = channeling_stats or {}
    ch_avg = _safe_float(channeling_stats.get("average", 0.0), 0.0)
    ch_max = _safe_float(channeling_stats.get("max", 0.0), 0.0)
    ch_min = _safe_float(channeling_stats.get("min", 0.0), 0.0)

    ch_range = ch_max - ch_min
    ch_range_norm = (ch_range / ch_max) if ch_max > 0 else 0.0
    if ch_range > 0:
        ch_coverage_norm = (ch_avg - ch_min) / ch_range
        ch_coverage_norm = max(0.0, min(1.0, ch_coverage_norm))
    else:
        ch_coverage_norm = 0.5

    return ch_range, ch_range_norm, ch_coverage_norm


def _video_sort_key(video_id):
    text = str(video_id).strip()
    if text.isdigit():
        return (0, int(text))
    return (1, text.lower())


def _prefix_row_features(row, prefix):
    features = {}
    for key, value in row.items():
        if key in identifier_columns:
            continue
        features[f"{prefix}:{key}"] = _safe_float_or_nan(value)
    return features


def _drop_sparse_columns(X, feature_columns, min_non_nan=2):
    keep_indices = []
    for index in range(X.shape[1]):
        if np.isfinite(X[:, index]).sum() >= min_non_nan:
            keep_indices.append(index)

    if not keep_indices:
        raise ValueError("No usable feature columns remain after filtering sparse values.")

    filtered_X = X[:, keep_indices]
    filtered_columns = [feature_columns[index] for index in keep_indices]
    return filtered_X, filtered_columns


def _build_feature_context_from_dataset(summary_dataset, selected_video_ids):
    if not summary_dataset:
        return {}

    feature_columns = list(summary_dataset.get("feature_columns") or [])
    if "summary:blonding_rate" not in feature_columns:
        return {}

    selected_ids = [str(video_id) for video_id in (selected_video_ids or [])]
    if not selected_ids:
        return {}

    video_ids = [str(video_id) for video_id in (summary_dataset.get("video_ids") or [])]
    id_to_index = {video_id: index for index, video_id in enumerate(video_ids)}
    raw_index = feature_columns.index("summary:blonding_rate")
    blonding_values = []
    for video_id in selected_ids:
        row_index = id_to_index.get(video_id)
        if row_index is None:
            continue
        raw_value = _safe_float_or_nan(summary_dataset["X"][row_index, raw_index])
        if not np.isnan(raw_value):
            blonding_values.append(float(raw_value))
    if not blonding_values:
        return {}

    return {
        "summary:blonding_rate_norm": {
            "type": "min_max_from_raw",
            "source_feature": "summary:blonding_rate",
            "min": float(min(blonding_values)),
            "max": float(max(blonding_values)),
        }
    }


def _read_labeled_feature_table(csv_path, prefix):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data not found: {csv_path}")

    row_map = {}
    fieldnames = None

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file_ref:
        reader = csv.DictReader(file_ref)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            video_id = str(row.get("video_id", "")).strip()
            label = _safe_int(row.get("label"), default=None)
            if not video_id or label not in (0, 1, 2):
                continue
            row_map[video_id] = {
                "label": label,
                "features": _prefix_row_features(row, prefix=prefix),
            }

    feature_columns = [
        f"{prefix}:{column_name}"
        for column_name in fieldnames
        if column_name not in identifier_columns
    ]
    return row_map, feature_columns


def _assemble_feature_dataset(summary_rows, summary_columns, events_rows, events_columns, feature_set):
    feature_key = str(feature_set).strip().lower()
    if feature_key not in supported_feature_sets:
        raise ValueError(
            f"Unsupported feature_set '{feature_set}'. "
            f"Choose one of: {supported_feature_sets}"
        )

    if feature_key == "summary":
        if not summary_rows:
            raise ValueError("Summary training data is empty.")
        ordered_ids = sorted(summary_rows.keys(), key=_video_sort_key)
        feature_columns = list(summary_columns)
        X_rows = [
            [summary_rows[video_id]["features"].get(column_name, np.nan) for column_name in feature_columns]
            for video_id in ordered_ids
        ]
        y_rows = [summary_rows[video_id]["label"] for video_id in ordered_ids]
    elif feature_key == "events":
        if not events_rows:
            raise ValueError("Events training data is empty.")
        ordered_ids = sorted(events_rows.keys(), key=_video_sort_key)
        feature_columns = list(events_columns)
        X_rows = [
            [events_rows[video_id]["features"].get(column_name, np.nan) for column_name in feature_columns]
            for video_id in ordered_ids
        ]
        y_rows = [events_rows[video_id]["label"] for video_id in ordered_ids]
    else:
        common_ids = sorted(set(summary_rows.keys()) & set(events_rows.keys()), key=_video_sort_key)
        if not common_ids:
            raise ValueError("Combined feature set needs both summary and event rows with matching video IDs.")

        y_rows = []
        X_rows = []
        feature_columns = list(summary_columns) + list(events_columns)
        for video_id in common_ids:
            summary_label = summary_rows[video_id]["label"]
            event_label = events_rows[video_id]["label"]
            if summary_label != event_label:
                raise ValueError(f"Mismatched labels for video '{video_id}' between summary and events CSV.")

            combined_features = {}
            combined_features.update(summary_rows[video_id]["features"])
            combined_features.update(events_rows[video_id]["features"])
            X_rows.append([combined_features.get(column_name, np.nan) for column_name in feature_columns])
            y_rows.append(summary_label)
        ordered_ids = common_ids

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int64)
    X, feature_columns = _drop_sparse_columns(X, feature_columns)
    return {
        "feature_set": feature_key,
        "video_ids": ordered_ids,
        "feature_columns": feature_columns,
        "X": X,
        "y": y,
    }


def _filter_dataset_feature_columns(dataset, selected_columns, feature_set_name=None):
    selected = [column_name for column_name in selected_columns if column_name in dataset["feature_columns"]]
    if not selected:
        raise ValueError("No requested feature columns were found in the dataset.")

    index_map = {column_name: index for index, column_name in enumerate(dataset["feature_columns"])}
    selected_indices = [index_map[column_name] for column_name in selected]
    return {
        "feature_set": feature_set_name or dataset["feature_set"],
        "video_ids": list(dataset["video_ids"]),
        "feature_columns": list(selected),
        "X": dataset["X"][:, selected_indices],
        "y": dataset["y"].copy(),
    }


def _load_training_datasets(summary_csv_path, events_csv_path):
    summary_rows, summary_columns = _read_labeled_feature_table(summary_csv_path, prefix="summary")
    events_rows, events_columns = _read_labeled_feature_table(events_csv_path, prefix="events")

    datasets = {}
    for feature_set in ("summary", "events", "combined"):
        try:
            datasets[feature_set] = _assemble_feature_dataset(
                summary_rows=summary_rows,
                summary_columns=summary_columns,
                events_rows=events_rows,
                events_columns=events_columns,
                feature_set=feature_set,
            )
        except ValueError:
            continue

    if not datasets:
        raise ValueError("No usable labeled training rows were found in the exported CSV files.")

    if "summary" in datasets:
        try:
            datasets["summary_compact"] = _filter_dataset_feature_columns(
                datasets["summary"],
                summary_compact_feature_columns,
                feature_set_name="summary_compact",
            )
        except ValueError:
            pass
        try:
            datasets["summary_compact_best"] = _filter_dataset_feature_columns(
                datasets["summary"],
                summary_compact_best_feature_columns,
                feature_set_name="summary_compact_best",
            )
        except ValueError:
            pass
    return datasets


def _subset_dataset_by_video_ids(dataset, selected_video_ids):
    selected_ids = [str(video_id) for video_id in (selected_video_ids or [])]
    index_by_id = {str(video_id): index for index, video_id in enumerate(dataset["video_ids"])}
    subset_indices = [index_by_id[video_id] for video_id in selected_ids if video_id in index_by_id]
    return {
        "feature_set": dataset["feature_set"],
        "video_ids": [dataset["video_ids"][index] for index in subset_indices],
        "feature_columns": list(dataset["feature_columns"]),
        "X": dataset["X"][subset_indices],
        "y": dataset["y"][subset_indices],
    }


def _align_candidate_datasets(datasets, candidate_feature_sets):
    reference_dataset = datasets[candidate_feature_sets[0]]
    common_ids = set(str(video_id) for video_id in reference_dataset["video_ids"])
    for feature_name in candidate_feature_sets[1:]:
        common_ids &= set(str(video_id) for video_id in datasets[feature_name]["video_ids"])

    ordered_ids = [
        str(video_id)
        for video_id in reference_dataset["video_ids"]
        if str(video_id) in common_ids
    ]
    if len(ordered_ids) < 2:
        raise ValueError("Need at least 2 shared labeled videos across the selected feature sets.")

    return {
        feature_name: _subset_dataset_by_video_ids(datasets[feature_name], ordered_ids)
        for feature_name in candidate_feature_sets
    }


def _normalise_test_size(test_size, sample_count, class_count):
    if 0 < float(test_size) < 1:
        test_count = int(math.ceil(float(sample_count) * float(test_size)))
    else:
        test_count = int(round(float(test_size)))

    test_count = max(int(class_count), test_count)
    test_count = min(int(sample_count - class_count), test_count)
    if test_count < int(class_count):
        raise ValueError(
            "Unable to create a stratified holdout split. "
            "Need enough samples to place at least one item from each class in both train and test."
        )
    return int(test_count)


def _build_holdout_split(dataset, test_size=default_test_size, random_state=default_random_state):
    video_ids = [str(video_id) for video_id in dataset["video_ids"]]
    y = np.array(dataset["y"], dtype=np.int64)
    class_counts = Counter(y.tolist())
    if len(class_counts) < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    min_class_count = min(class_counts.values())
    if min_class_count < 2:
        raise ValueError(
            "A proper stratified holdout split needs at least 2 samples in every class. "
            f"Observed class counts: {dict(class_counts)}"
        )

    test_count = _normalise_test_size(test_size, sample_count=len(video_ids), class_count=len(class_counts))
    train_ids, test_ids = train_test_split(
        video_ids,
        test_size=test_count,
        stratify=y,
        random_state=random_state,
    )

    label_by_video_id = {
        str(video_id): int(dataset["y"][index])
        for index, video_id in enumerate(dataset["video_ids"])
    }
    train_counter = Counter(label_by_video_id[video_id] for video_id in train_ids)
    test_counter = Counter(label_by_video_id[video_id] for video_id in test_ids)

    return {
        "train_video_ids": [str(video_id) for video_id in train_ids],
        "test_video_ids": [str(video_id) for video_id in test_ids],
        "train_size": len(train_ids),
        "test_size": len(test_ids),
        "class_counts_full": {int(key): int(value) for key, value in class_counts.items()},
        "class_counts_train": {int(key): int(value) for key, value in train_counter.items()},
        "class_counts_test": {int(key): int(value) for key, value in test_counter.items()},
        "test_fraction": float(len(test_ids) / max(1, len(video_ids))),
        "random_state": int(random_state),
        "stratified": True,
    }


def build_model(model_type="svm", random_state=42):
    model_key = str(model_type).strip().lower()
    if model_key not in supported_model_types:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Choose one of: {supported_model_types}"
        )

    if model_key == "auto":
        raise ValueError("build_model() requires a concrete model type, not 'auto'.")

    if model_key == "svm":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_key == "logistic":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.5,
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=5000,
                    ),
                ),
            ]
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def _candidate_model_types(model_type):
    model_key = str(model_type).strip().lower()
    if model_key not in supported_model_types:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Choose one of: {supported_model_types}"
        )
    if model_key == "all":
        return ["logistic", "svm", "random_forest"]
    if model_key == "both":
        return ["svm", "random_forest"]
    return [model_key]


def _candidate_feature_sets(feature_set, datasets):
    feature_key = str(feature_set).strip().lower()
    if feature_key not in supported_feature_sets:
        raise ValueError(
            f"Unsupported feature_set '{feature_set}'. "
            f"Choose one of: {supported_feature_sets}"
        )
    if feature_key == "auto":
        return [
            name
            for name in ("summary_compact_best", "summary_compact", "summary", "events", "combined")
            if name in datasets
        ]
    if feature_key not in datasets:
        raise ValueError(f"Feature set '{feature_key}' is unavailable with the provided CSV files.")
    return [feature_key]


def _run_cross_validation(model, X, y, cv):
    return cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=evaluation_scoring,
        error_score="raise",
        n_jobs=None,
    )


def _apply_cv_metrics(metrics, cv_results):
    metrics["cv_accuracy_mean"] = float(np.mean(cv_results["test_accuracy"]))
    metrics["cv_accuracy_std"] = float(np.std(cv_results["test_accuracy"]))
    metrics["cv_balanced_accuracy_mean"] = float(np.mean(cv_results["test_balanced_accuracy"]))
    metrics["cv_balanced_accuracy_std"] = float(np.std(cv_results["test_balanced_accuracy"]))
    metrics["cv_macro_f1_mean"] = float(np.mean(cv_results["test_macro_f1"]))
    metrics["cv_macro_f1_std"] = float(np.std(cv_results["test_macro_f1"]))
    metrics["cv_ideal_vs_nonideal_accuracy_mean"] = float(np.mean(cv_results["test_ideal_vs_nonideal_accuracy"]))
    metrics["cv_ideal_vs_nonideal_accuracy_std"] = float(np.std(cv_results["test_ideal_vs_nonideal_accuracy"]))
    metrics["cv_ideal_vs_nonideal_balanced_accuracy_mean"] = float(
        np.mean(cv_results["test_ideal_vs_nonideal_balanced_accuracy"])
    )
    metrics["cv_ideal_vs_nonideal_balanced_accuracy_std"] = float(
        np.std(cv_results["test_ideal_vs_nonideal_balanced_accuracy"])
    )
    metrics["cv_ideal_vs_nonideal_macro_f1_mean"] = float(np.mean(cv_results["test_ideal_vs_nonideal_macro_f1"]))
    metrics["cv_ideal_vs_nonideal_macro_f1_std"] = float(np.std(cv_results["test_ideal_vs_nonideal_macro_f1"]))


def _evaluate_candidate(model, X, y, random_state=42):
    class_counts = Counter(y.tolist())
    metrics = {
        "evaluation_mode": "train_only",
        "cv_splits": None,
        "cv_repeats": None,
        "training_accuracy": None,
        "cv_accuracy_mean": None,
        "cv_accuracy_std": None,
        "cv_balanced_accuracy_mean": None,
        "cv_balanced_accuracy_std": None,
        "cv_macro_f1_mean": None,
        "cv_macro_f1_std": None,
        "cv_ideal_vs_nonideal_accuracy_mean": None,
        "cv_ideal_vs_nonideal_accuracy_std": None,
        "cv_ideal_vs_nonideal_balanced_accuracy_mean": None,
        "cv_ideal_vs_nonideal_balanced_accuracy_std": None,
        "cv_ideal_vs_nonideal_macro_f1_mean": None,
        "cv_ideal_vs_nonideal_macro_f1_std": None,
        "training_ideal_vs_nonideal_accuracy": None,
        "classification_report": None,
    }

    min_class_count = min(class_counts.values())
    if len(y) >= 12 and min_class_count >= 3:
        n_repeats = 5 if len(y) >= 18 else 2
        cv = RepeatedStratifiedKFold(
            n_splits=3,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        cv_results = _run_cross_validation(model, X, y, cv)
        metrics["evaluation_mode"] = "repeated_stratified_kfold"
        metrics["cv_splits"] = 3
        metrics["cv_repeats"] = n_repeats
        _apply_cv_metrics(metrics, cv_results)
    elif len(y) >= 6 and min_class_count >= 2:
        n_splits = min(5, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_results = _run_cross_validation(model, X, y, cv)
        metrics["evaluation_mode"] = "stratified_kfold"
        metrics["cv_splits"] = n_splits
        _apply_cv_metrics(metrics, cv_results)

    model.fit(X, y)
    y_pred = model.predict(X)
    metrics["training_accuracy"] = float(accuracy_score(y, y_pred))
    metrics["classification_report"] = classification_report(
        y,
        y_pred,
        labels=[0, 1, 2],
        target_names=["under", "ideal", "over"],
        zero_division=0,
    )
    metrics["training_ideal_vs_nonideal_accuracy"] = float(_ideal_vs_nonideal_accuracy(y, y_pred))
    return metrics


def _prediction_confidences(model, X):
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X)
            return [float(np.max(row)) for row in probabilities]
        except Exception:
            return None
    return None


def _evaluate_holdout(model, X_train, y_train, X_test, y_test, test_video_ids=None):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    confidences = _prediction_confidences(model, X_test)
    y_train_binary = _ideal_binary_labels(y_train)
    y_test_binary = _ideal_binary_labels(y_test)
    y_train_pred_binary = _ideal_binary_labels(y_train_pred)
    y_test_pred_binary = _ideal_binary_labels(y_test_pred)
    ordered_test_ids = [str(video_id) for video_id in (test_video_ids or [])]
    test_predictions = []
    for index, predicted_class in enumerate(y_test_pred.tolist()):
        true_class = int(y_test[index])
        row = {
            "video_id": ordered_test_ids[index] if index < len(ordered_test_ids) else str(index),
            "true_class": true_class,
            "true_label": class_name_map.get(true_class, str(true_class)),
            "predicted_class": int(predicted_class),
            "predicted_label": class_name_map.get(int(predicted_class), str(predicted_class)),
            "true_is_ideal": bool(true_class == 1),
            "predicted_is_ideal": bool(int(predicted_class) == 1),
        }
        if confidences is not None and index < len(confidences):
            row["confidence"] = float(confidences[index])
        test_predictions.append(row)

    return {
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "test_macro_f1": float(f1_score(y_test, y_test_pred, average="macro", zero_division=0)),
        "train_ideal_vs_nonideal_accuracy": float(accuracy_score(y_train_binary, y_train_pred_binary)),
        "test_ideal_vs_nonideal_accuracy": float(accuracy_score(y_test_binary, y_test_pred_binary)),
        "test_ideal_vs_nonideal_balanced_accuracy": float(
            balanced_accuracy_score(y_test_binary, y_test_pred_binary)
        ),
        "test_ideal_vs_nonideal_macro_f1": float(
            f1_score(y_test_binary, y_test_pred_binary, average="macro", zero_division=0)
        ),
        "test_classification_report": classification_report(
            y_test,
            y_test_pred,
            labels=[0, 1, 2],
            target_names=["under", "ideal", "over"],
            zero_division=0,
        ),
        "test_ideal_vs_nonideal_confusion_matrix": confusion_matrix(
            y_test_binary,
            y_test_pred_binary,
            labels=[0, 1],
        ).tolist(),
        "test_confusion_matrix": confusion_matrix(
            y_test,
            y_test_pred,
            labels=[0, 1, 2],
        ).tolist(),
        "test_predictions": test_predictions,
    }


def _candidate_rank_key(result):
    metrics = result["metrics"]
    score = metrics.get("cv_balanced_accuracy_mean")
    macro_f1 = metrics.get("cv_macro_f1_mean")
    accuracy = metrics.get("cv_accuracy_mean")
    ideal_nonideal_accuracy = metrics.get("cv_ideal_vs_nonideal_accuracy_mean")
    return (
        -1.0 if score is None else float(score),
        -1.0 if macro_f1 is None else float(macro_f1),
        -1.0 if ideal_nonideal_accuracy is None else float(ideal_nonideal_accuracy),
        -1.0 if accuracy is None else float(accuracy),
        float(metrics.get("training_accuracy") or 0.0),
    )


def _serialise_candidate_result(result):
    return {
        "model_type": result["model_type"],
        "feature_set": result["feature_set"],
        "sample_count": int(result["sample_count"]),
        "feature_count": int(result["feature_count"]),
        "metrics": result["metrics"],
    }


def _format_metric(value):
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _print_training_header(summary_csv_path, events_csv_path, class_counts, feature_set, model_type, test_size, random_state):
    print("\n" + "=" * 60)
    print("TRAINING EXTRACTION LEVEL MODEL")
    print("=" * 60)
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Events CSV: {events_csv_path}")
    print(f"Class counts: {dict(class_counts)}")
    print(f"Feature set request: {feature_set}")
    print(f"Model type request: {model_type}")
    print(f"Holdout test fraction request: {float(test_size):.3f}")
    print(f"Random state: {int(random_state)}")


def _evaluate_training_candidates(datasets, candidate_feature_sets, candidate_model_types, train_video_ids, random_state=42):
    evaluated_candidates = []

    for feature_name in candidate_feature_sets:
        dataset = _subset_dataset_by_video_ids(datasets[feature_name], train_video_ids)
        X = dataset["X"]
        y = dataset["y"]

        print(
            f"\nFeature set '{feature_name}': {len(y)} samples, "
            f"{len(dataset['feature_columns'])} usable features"
        )

        for model_name in candidate_model_types:
            print(f"Evaluating {feature_name} + {model_name} ...")
            model = build_model(model_type=model_name, random_state=random_state)
            metrics = _evaluate_candidate(model, X, y, random_state=random_state)
            result = {
                "feature_set": feature_name,
                "model_type": model_name,
                "sample_count": len(y),
                "feature_count": len(dataset["feature_columns"]),
                "metrics": metrics,
            }
            evaluated_candidates.append(result)
            print(
                "  balanced_acc="
                f"{_format_metric(metrics.get('cv_balanced_accuracy_mean'))}, "
                "macro_f1="
                f"{_format_metric(metrics.get('cv_macro_f1_mean'))}, "
                "train_acc="
                f"{_format_metric(metrics.get('training_accuracy'))}"
            )

    return evaluated_candidates


def _fit_best_model_on_training_split(best_result, datasets, train_video_ids, test_video_ids, random_state=42):
    best_dataset = datasets[best_result["feature_set"]]
    train_dataset = _subset_dataset_by_video_ids(best_dataset, train_video_ids)
    test_dataset = _subset_dataset_by_video_ids(best_dataset, test_video_ids)
    best_model = build_model(model_type=best_result["model_type"], random_state=random_state)
    holdout_metrics = _evaluate_holdout(
        best_model,
        train_dataset["X"],
        train_dataset["y"],
        test_dataset["X"],
        test_dataset["y"],
        test_video_ids=test_dataset["video_ids"],
    )
    return best_model, train_dataset, test_dataset, holdout_metrics


def _build_model_bundle(
    best_result,
    best_model,
    summary_csv_path,
    events_csv_path,
    feature_context,
    split_info,
    train_dataset,
    test_dataset,
    holdout_metrics,
):
    return {
        "model": best_model,
        "model_type": best_result["model_type"],
        "feature_set": best_result["feature_set"],
        "feature_columns": train_dataset["feature_columns"],
        "class_name_map": class_name_map,
        "metrics": best_result["metrics"],
        "feature_context": feature_context,
        "training_summary": {
            "summary_csv_path": summary_csv_path,
            "events_csv_path": events_csv_path,
            "sample_count": len(train_dataset["y"]),
            "feature_count": len(train_dataset["feature_columns"]),
            "class_counts": dict(Counter(train_dataset["y"].tolist())),
            "training_video_ids": list(train_dataset["video_ids"]),
        },
        "evaluation": {
            "protocol": "heldout_test_plus_cross_validation_model_selection",
            "selection_metric": "cv_balanced_accuracy_mean",
            "selection_train_only_metrics": best_result["metrics"],
            "holdout_metrics": holdout_metrics,
            "split": split_info,
            "test_summary": {
                "sample_count": len(test_dataset["y"]),
                "class_counts": dict(Counter(test_dataset["y"].tolist())),
                "test_video_ids": list(test_dataset["video_ids"]),
            },
        },
    }


def _save_json_artifact(output_path, payload):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_ref:
        json.dump(payload, file_ref, indent=2)


def _artifact_paths_from_model_output(model_output_path):
    root, ext = os.path.splitext(model_output_path)
    if not ext:
        root = model_output_path
    return {
        "evaluation_json": f"{root}_evaluation.json",
        "split_json": f"{root}_split.json",
    }


def _save_model_bundle(bundle, model_output_path, candidate_results):
    bundle_to_save = dict(bundle)
    bundle_to_save["candidate_results"] = [
        _serialise_candidate_result(result) for result in candidate_results
    ]
    bundle_to_save["artifact_paths"] = _artifact_paths_from_model_output(model_output_path)
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(bundle_to_save, model_output_path)

    artifact_paths = bundle_to_save["artifact_paths"]
    _save_json_artifact(
        artifact_paths["evaluation_json"],
        {
            "model_type": bundle_to_save.get("model_type"),
            "feature_set": bundle_to_save.get("feature_set"),
            "metrics": bundle_to_save.get("metrics"),
            "evaluation": bundle_to_save.get("evaluation"),
            "candidate_results": bundle_to_save.get("candidate_results"),
        },
    )
    _save_json_artifact(
        artifact_paths["split_json"],
        bundle_to_save.get("evaluation", {}).get("split", {}),
    )
    return bundle_to_save


def _print_best_model_summary(best_result, bundle, model_output_path):
    metrics = best_result["metrics"]
    evaluation = bundle.get("evaluation") or {}
    holdout_metrics = evaluation.get("holdout_metrics") or {}
    print("\n" + "-" * 60)
    print("BEST MODEL")
    print("-" * 60)
    print(f"Selected feature set: {best_result['feature_set']}")
    print(f"Selected model type: {best_result['model_type']}")
    print(f"Evaluation mode: {metrics['evaluation_mode']}")
    if metrics["cv_balanced_accuracy_mean"] is not None:
        print(
            "CV balanced accuracy: "
            f"{metrics['cv_balanced_accuracy_mean']:.4f} "
            f"+/- {metrics['cv_balanced_accuracy_std']:.4f}"
        )
    if metrics["cv_macro_f1_mean"] is not None:
        print(
            "CV macro F1: "
            f"{metrics['cv_macro_f1_mean']:.4f} "
            f"+/- {metrics['cv_macro_f1_std']:.4f}"
        )
    if metrics.get("cv_ideal_vs_nonideal_accuracy_mean") is not None:
        print(
            "CV ideal-vs-nonideal accuracy: "
            f"{metrics['cv_ideal_vs_nonideal_accuracy_mean']:.4f} "
            f"+/- {metrics['cv_ideal_vs_nonideal_accuracy_std']:.4f}"
        )
    print(f"Training accuracy (selection fit): {metrics['training_accuracy']:.4f}")
    if metrics.get("training_ideal_vs_nonideal_accuracy") is not None:
        print(
            "Training ideal-vs-nonideal accuracy (selection fit): "
            f"{metrics['training_ideal_vs_nonideal_accuracy']:.4f}"
        )
    if holdout_metrics:
        print(f"Holdout accuracy: {holdout_metrics['test_accuracy']:.4f}")
        print(f"Holdout balanced accuracy: {holdout_metrics['test_balanced_accuracy']:.4f}")
        print(f"Holdout macro F1: {holdout_metrics['test_macro_f1']:.4f}")
        print(
            "Holdout ideal-vs-nonideal accuracy: "
            f"{holdout_metrics['test_ideal_vs_nonideal_accuracy']:.4f}"
        )
    print(f"Model saved to: {model_output_path}")
    artifact_paths = bundle.get("artifact_paths") or {}
    if artifact_paths.get("evaluation_json"):
        print(f"Evaluation report: {artifact_paths['evaluation_json']}")
    if artifact_paths.get("split_json"):
        print(f"Split manifest: {artifact_paths['split_json']}")
    print("=" * 60)


def train_model(
    summary_csv_path=default_summary_csv,
    events_csv_path=default_events_csv,
    model_output_path=default_model_path,
    random_state=default_random_state,
    model_type=model_type_default,
    feature_set=feature_set_default,
    test_size=default_test_size,
):
    datasets = _load_training_datasets(summary_csv_path, events_csv_path)
    candidate_feature_sets = _candidate_feature_sets(feature_set, datasets)
    candidate_model_types = _candidate_model_types(model_type)
    aligned_datasets = _align_candidate_datasets(datasets, candidate_feature_sets)

    reference_dataset = aligned_datasets[candidate_feature_sets[0]]
    class_counts = Counter(reference_dataset["y"].tolist())

    _print_training_header(
        summary_csv_path=summary_csv_path,
        events_csv_path=events_csv_path,
        class_counts=class_counts,
        feature_set=feature_set,
        model_type=model_type,
        test_size=test_size,
        random_state=random_state,
    )

    if len(class_counts) < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    split_info = _build_holdout_split(
        reference_dataset,
        test_size=test_size,
        random_state=random_state,
    )
    print(
        "Holdout split: "
        f"train={split_info['train_size']} "
        f"test={split_info['test_size']} "
        f"test_fraction={split_info['test_fraction']:.3f}"
    )

    evaluated_candidates = _evaluate_training_candidates(
        datasets=aligned_datasets,
        candidate_feature_sets=candidate_feature_sets,
        candidate_model_types=candidate_model_types,
        train_video_ids=split_info["train_video_ids"],
        random_state=random_state,
    )

    if not evaluated_candidates:
        raise ValueError("No candidate models could be evaluated.")

    best_result = max(evaluated_candidates, key=_candidate_rank_key)
    best_model, train_dataset, test_dataset, holdout_metrics = _fit_best_model_on_training_split(
        best_result=best_result,
        datasets=aligned_datasets,
        train_video_ids=split_info["train_video_ids"],
        test_video_ids=split_info["test_video_ids"],
        random_state=random_state,
    )
    feature_context = _build_feature_context_from_dataset(
        aligned_datasets.get("summary"),
        split_info["train_video_ids"],
    )
    bundle = _build_model_bundle(
        best_result=best_result,
        best_model=best_model,
        summary_csv_path=summary_csv_path,
        events_csv_path=events_csv_path,
        feature_context=feature_context,
        split_info=split_info,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        holdout_metrics=holdout_metrics,
    )
    bundle = _save_model_bundle(bundle, model_output_path, evaluated_candidates)
    _print_best_model_summary(best_result, bundle, model_output_path)

    return bundle


def _multi_model_output_paths(model_output_path=None):
    base_path = model_output_path or default_model_path
    if os.path.normcase(base_path) == os.path.normcase(default_model_path):
        return {
            "random_forest": default_rf_model_path,
            "svm": default_svm_model_path,
            "logistic": default_logistic_model_path,
        }

    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".joblib"
    return {
        "random_forest": f"{root}_rf{ext}",
        "svm": f"{root}_svm{ext}",
        "logistic": f"{root}_logistic{ext}",
    }


def _print_model_set_header(requested_models, feature_set):
    print("\n" + "=" * 60)
    print("TRAINING MODEL SET")
    print("=" * 60)
    print(f"Requested models: {', '.join(requested_models)}")
    print(f"Feature set request: {feature_set}")


def _print_model_comparison(requested_models, bundles, output_paths):
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    for single_model_type in requested_models:
        bundle = bundles[single_model_type]
        metrics = bundle.get("metrics") or {}
        print(
            f"{single_model_type}: "
            f"feature_set={bundle.get('feature_set')}, "
            f"cv_bal_acc={_format_metric(metrics.get('cv_balanced_accuracy_mean'))}, "
            f"cv_ideal_nonideal_acc={_format_metric(metrics.get('cv_ideal_vs_nonideal_accuracy_mean'))}, "
            f"holdout_bal_acc={_format_metric((bundle.get('evaluation') or {}).get('holdout_metrics', {}).get('test_balanced_accuracy'))}, "
            f"holdout_ideal_nonideal_acc={_format_metric((bundle.get('evaluation') or {}).get('holdout_metrics', {}).get('test_ideal_vs_nonideal_accuracy'))}, "
            f"cv_macro_f1={_format_metric(metrics.get('cv_macro_f1_mean'))}, "
            f"train_acc={_format_metric(metrics.get('training_accuracy'))}, "
            f"path={output_paths[single_model_type]}"
        )
    print("=" * 60)


def train_models(
    summary_csv_path=default_summary_csv,
    events_csv_path=default_events_csv,
    model_output_path=default_model_path,
    random_state=default_random_state,
    model_type=model_type_default,
    feature_set=feature_set_default,
    test_size=default_test_size,
):
    requested_models = _candidate_model_types(model_type)
    output_paths = _multi_model_output_paths(model_output_path)
    bundles = {}

    _print_model_set_header(requested_models, feature_set)

    for single_model_type in requested_models:
        print("\n" + "#" * 60)
        print(f"TRAINING {single_model_type.upper()}")
        print("#" * 60)
        bundle = train_model(
            summary_csv_path=summary_csv_path,
            events_csv_path=events_csv_path,
            model_output_path=output_paths[single_model_type],
            random_state=random_state,
            model_type=single_model_type,
            feature_set=feature_set,
            test_size=test_size,
        )
        bundles[single_model_type] = bundle

    if len(bundles) > 1:
        _print_model_comparison(requested_models, bundles, output_paths)

    return bundles


def load_model_bundle(model_path=default_model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def _build_feature_rows_from_results(results_dict):
    frame_rows = build_frame_rows(results_dict, label=None)
    event_row = build_event_row(results_dict, label=None, frame_rows=frame_rows)
    summary_row = build_summary_row(results_dict, label=None, event_row=event_row)
    return {
        "summary": _prefix_row_features(summary_row, prefix="summary"),
        "events": _prefix_row_features(event_row, prefix="events"),
    }


def _merge_feature_rows(*rows):
    combined = {}
    for row in rows:
        combined.update(row)
    return combined


def feature_row_from_results_dict(results_dict, feature_set="summary"):
    feature_key = str(feature_set).strip().lower()
    if feature_key == "legacy":
        flow_start = _safe_int(results_dict.get("flow_start"), default=0) or 0
        flow_end = _safe_int(results_dict.get("flow_end"), default=0) or 0
        fps = safe_fps(results_dict.get("fps"), default=DEFAULT_EXTRACTION_FPS)
        shot_time = shot_duration_seconds(flow_start, flow_end, fps)
        blonding_rate = _safe_float(results_dict.get("blond_rate"), default=0.0)
        ch_range, ch_range_norm, ch_coverage_norm = _compute_channeling_metrics(
            results_dict.get("channeling_stats")
        )
        return {
            "shot_time": shot_time,
            "blonding_rate": blonding_rate,
            "channeling_range": ch_range,
            "channeling_range_norm": ch_range_norm,
            "channeling_coverage_norm": ch_coverage_norm,
        }

    feature_rows = _build_feature_rows_from_results(results_dict)
    if feature_key == "summary":
        return feature_rows["summary"]
    if feature_key == "summary_compact":
        return {
            column_name: feature_rows["summary"].get(column_name)
            for column_name in summary_compact_feature_columns
        }
    if feature_key == "summary_compact_best":
        return {
            column_name: feature_rows["summary"].get(column_name)
            for column_name in summary_compact_best_feature_columns
        }
    if feature_key == "events":
        return feature_rows["events"]
    if feature_key == "combined":
        return _merge_feature_rows(feature_rows["summary"], feature_rows["events"])

    raise ValueError(
        f"Unsupported feature_set '{feature_set}'. "
        f"Choose one of: {supported_feature_sets}"
    )


def feature_row_from_results_json(results_json_path, feature_set="summary"):
    if not os.path.exists(results_json_path):
        raise FileNotFoundError(f"Results JSON not found: {results_json_path}")
    with open(results_json_path, "r", encoding="utf-8") as file_ref:
        data = json.load(file_ref)
    return feature_row_from_results_dict(data, feature_set=feature_set), data


def _apply_bundle_feature_context(feature_row, bundle):
    context = bundle.get("feature_context") or {}
    hydrated = dict(feature_row)

    for target_feature, config in context.items():
        if target_feature in hydrated and hydrated[target_feature] is not None and not np.isnan(
            _safe_float_or_nan(hydrated[target_feature])
        ):
            continue

        if config.get("type") == "min_max_from_raw":
            source_feature = config.get("source_feature")
            raw_value = _safe_float_or_nan(hydrated.get(source_feature))
            if np.isnan(raw_value):
                continue
            min_value = _safe_float(config.get("min"), default=raw_value)
            max_value = _safe_float(config.get("max"), default=raw_value)
            if abs(max_value - min_value) < 1e-9:
                hydrated[target_feature] = 0.5
            else:
                hydrated[target_feature] = (raw_value - min_value) / (max_value - min_value)

    return hydrated


def _predict_from_bundle(feature_row, bundle):
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", list(legacy_feature_columns))
    names_map = bundle.get("class_name_map", class_name_map)

    X = np.array(
        [[_safe_float(feature_row.get(column_name), default=0.0) for column_name in feature_columns]],
        dtype=np.float32,
    )
    predicted_class = int(model.predict(X)[0])

    confidence = None
    class_probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        classes = getattr(model, "classes_", np.arange(len(proba)))
        class_probabilities = {
            names_map.get(int(class_id), str(class_id)): float(probability)
            for class_id, probability in zip(classes, proba)
        }

    return {
        "predicted_class": predicted_class,
        "predicted_label": names_map.get(predicted_class, str(predicted_class)),
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "feature_vector": {
            column_name: _safe_float(feature_row.get(column_name), default=0.0)
            for column_name in feature_columns
        },
    }


def _resolve_bundle_feature_set(bundle):
    bundle_feature_set = bundle.get("feature_set")
    if bundle_feature_set is None:
        return "legacy"
    return bundle_feature_set


def predict_from_feature_row(feature_row, model_path=default_model_path):
    bundle = load_model_bundle(model_path)
    return _predict_from_bundle(feature_row, bundle)


def predict_from_results_json(results_json_path, model_path=default_model_path):
    bundle = load_model_bundle(model_path)
    feature_set = _resolve_bundle_feature_set(bundle)

    feature_row, raw_results = feature_row_from_results_json(
        results_json_path,
        feature_set=feature_set,
    )
    feature_row = _apply_bundle_feature_context(feature_row, bundle)
    prediction = _predict_from_bundle(feature_row, bundle)
    prediction["video_name"] = raw_results.get("video_name")
    prediction["results_json"] = results_json_path
    prediction["feature_set"] = feature_set
    prediction["model_type"] = bundle.get("model_type")
    return prediction


def predict_from_results_dict(results_dict, model_path=default_model_path):
    bundle = load_model_bundle(model_path)
    feature_set = _resolve_bundle_feature_set(bundle)
    feature_row = feature_row_from_results_dict(results_dict, feature_set=feature_set)
    feature_row = _apply_bundle_feature_context(feature_row, bundle)
    prediction = _predict_from_bundle(feature_row, bundle)
    prediction["video_name"] = results_dict.get("video_name")
    prediction["feature_set"] = feature_set
    prediction["model_type"] = bundle.get("model_type")
    return prediction


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Train and use espresso extraction classifier.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    train_parser = subparsers.add_parser(
        "train",
        help="Train model from exported training CSV files (default action when no command is given)",
    )
    train_parser.add_argument(
        "--csv",
        default=default_summary_csv,
        help="Path to summary training CSV (Analysis/training_data.csv)",
    )
    train_parser.add_argument(
        "--events-csv",
        default=default_events_csv,
        help="Path to event training CSV (Analysis/training_data_events.csv)",
    )
    train_parser.add_argument("--model", default=default_model_path, help="Path to save trained model")
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=default_test_size,
        help="Held-out test fraction (0-1) or absolute test count. Default: 0.2",
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=default_random_state,
        help="Random seed for the holdout split and CV reproducibility.",
    )
    train_parser.add_argument(
        "--model-type",
        default=model_type_default,
        choices=list(supported_model_types),
        help="Model family to train. Use 'all' to compare logistic, Random Forest, and SVM together; 'both' keeps the legacy Random Forest + SVM comparison.",
    )
    train_parser.add_argument(
        "--feature-set",
        default=feature_set_default,
        choices=list(supported_feature_sets),
        help="Which exported feature set to use, or 'auto' to compare the compact summary profiles alongside the broader exported sets.",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict extraction class from *_results.json")
    predict_parser.add_argument("--results-json", required=True, help="Path to analysis results JSON")
    predict_parser.add_argument("--model", default=default_model_path, help="Path to trained model")
    predict_parser.add_argument("--print-json", action="store_true", help="Print full JSON output")

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command is None:
        train_models(
            summary_csv_path=default_summary_csv,
            events_csv_path=default_events_csv,
            model_output_path=default_model_path,
            random_state=default_random_state,
            model_type=model_type_default,
            feature_set=feature_set_default,
            test_size=default_test_size,
        )
        return

    if args.command == "train":
        train_models(
            summary_csv_path=args.csv,
            events_csv_path=args.events_csv,
            model_output_path=args.model,
            random_state=args.random_state,
            model_type=args.model_type,
            feature_set=args.feature_set,
            test_size=args.test_size,
        )
        return

    if args.command == "predict":
        result = predict_from_results_json(args.results_json, model_path=args.model)
        if args.print_json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "=" * 60)
            print("EXTRACTION PREDICTION")
            print("=" * 60)
            print(f"Video: {result.get('video_name')}")
            print(f"Feature set: {result.get('feature_set')}")
            print(f"Model type: {result.get('model_type')}")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Predicted label: {result['predicted_label']}")
            if result["confidence"] is not None:
                print(f"Confidence: {result['confidence']:.4f}")
        return


if __name__ == "__main__":
    main()
