# Train and run extraction-level classifiers.

import argparse
import csv
import json
import math
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Data_Export import build_event_row, build_frame_rows, build_summary_row

base_dir = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.join(base_dir, "Analysis")

default_summary_csv = os.path.join(analysis_dir, "training_data.csv")
default_events_csv = os.path.join(analysis_dir, "training_data_events.csv")
default_model_path = os.path.join(analysis_dir, "extraction_model.joblib")

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
model_type_default = "auto"

supported_feature_sets = ("auto", "summary", "events", "combined")
supported_model_types = ("auto", "logistic_regression", "svm", "random_forest", "extra_trees")


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


def _load_feature_context(summary_csv_path):
    if not os.path.exists(summary_csv_path):
        return {}

    blonding_values = []
    with open(summary_csv_path, "r", encoding="utf-8-sig", newline="") as file_ref:
        reader = csv.DictReader(file_ref)
        for row in reader:
            label = _safe_int(row.get("label"), default=None)
            if label not in (0, 1, 2):
                continue
            blonding_values.append(_safe_float(row.get("blonding_rate"), default=0.0))

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
    return datasets


def build_model(model_type="logistic_regression", random_state=42):
    model_key = str(model_type).strip().lower()
    if model_key not in supported_model_types:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Choose one of: {supported_model_types}"
        )

    if model_key == "auto":
        raise ValueError("build_model() requires a concrete model type, not 'auto'.")

    if model_key == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )

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

    if model_key == "random_forest":
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

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    random_state=random_state,
                    class_weight="balanced",
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
    if model_key == "auto":
        return ["logistic_regression", "svm", "random_forest", "extra_trees"]
    return [model_key]


def _candidate_feature_sets(feature_set, datasets):
    feature_key = str(feature_set).strip().lower()
    if feature_key not in supported_feature_sets:
        raise ValueError(
            f"Unsupported feature_set '{feature_set}'. "
            f"Choose one of: {supported_feature_sets}"
        )
    if feature_key == "auto":
        return [name for name in ("summary", "events", "combined") if name in datasets]
    if feature_key not in datasets:
        raise ValueError(f"Feature set '{feature_key}' is unavailable with the provided CSV files.")
    return [feature_key]


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
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "macro_f1": "f1_macro",
            },
            error_score="raise",
            n_jobs=None,
        )
        metrics["evaluation_mode"] = "repeated_stratified_kfold"
        metrics["cv_splits"] = 3
        metrics["cv_repeats"] = n_repeats
        metrics["cv_accuracy_mean"] = float(np.mean(cv_results["test_accuracy"]))
        metrics["cv_accuracy_std"] = float(np.std(cv_results["test_accuracy"]))
        metrics["cv_balanced_accuracy_mean"] = float(np.mean(cv_results["test_balanced_accuracy"]))
        metrics["cv_balanced_accuracy_std"] = float(np.std(cv_results["test_balanced_accuracy"]))
        metrics["cv_macro_f1_mean"] = float(np.mean(cv_results["test_macro_f1"]))
        metrics["cv_macro_f1_std"] = float(np.std(cv_results["test_macro_f1"]))
    elif len(y) >= 6 and min_class_count >= 2:
        n_splits = min(5, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "macro_f1": "f1_macro",
            },
            error_score="raise",
            n_jobs=None,
        )
        metrics["evaluation_mode"] = "stratified_kfold"
        metrics["cv_splits"] = n_splits
        metrics["cv_accuracy_mean"] = float(np.mean(cv_results["test_accuracy"]))
        metrics["cv_accuracy_std"] = float(np.std(cv_results["test_accuracy"]))
        metrics["cv_balanced_accuracy_mean"] = float(np.mean(cv_results["test_balanced_accuracy"]))
        metrics["cv_balanced_accuracy_std"] = float(np.std(cv_results["test_balanced_accuracy"]))
        metrics["cv_macro_f1_mean"] = float(np.mean(cv_results["test_macro_f1"]))
        metrics["cv_macro_f1_std"] = float(np.std(cv_results["test_macro_f1"]))

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
    return metrics


def _candidate_rank_key(result):
    metrics = result["metrics"]
    score = metrics.get("cv_balanced_accuracy_mean")
    macro_f1 = metrics.get("cv_macro_f1_mean")
    accuracy = metrics.get("cv_accuracy_mean")
    return (
        -1.0 if score is None else float(score),
        -1.0 if macro_f1 is None else float(macro_f1),
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


def train_model(
    summary_csv_path=default_summary_csv,
    events_csv_path=default_events_csv,
    model_output_path=default_model_path,
    random_state=42,
    model_type=model_type_default,
    feature_set=feature_set_default,
):
    datasets = _load_training_datasets(summary_csv_path, events_csv_path)
    feature_context = _load_feature_context(summary_csv_path)
    candidate_feature_sets = _candidate_feature_sets(feature_set, datasets)
    candidate_model_types = _candidate_model_types(model_type)

    reference_dataset = datasets[candidate_feature_sets[0]]
    class_counts = Counter(reference_dataset["y"].tolist())

    print("\n" + "=" * 60)
    print("TRAINING EXTRACTION LEVEL MODEL")
    print("=" * 60)
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Events CSV: {events_csv_path}")
    print(f"Class counts: {dict(class_counts)}")
    print(f"Feature set request: {feature_set}")
    print(f"Model type request: {model_type}")

    if len(class_counts) < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    evaluated_candidates = []
    for feature_name in candidate_feature_sets:
        dataset = datasets[feature_name]
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

    if not evaluated_candidates:
        raise ValueError("No candidate models could be evaluated.")

    best_result = max(evaluated_candidates, key=_candidate_rank_key)
    best_dataset = datasets[best_result["feature_set"]]
    best_model = build_model(model_type=best_result["model_type"], random_state=random_state)
    best_model.fit(best_dataset["X"], best_dataset["y"])

    bundle = {
        "model": best_model,
        "model_type": best_result["model_type"],
        "feature_set": best_result["feature_set"],
        "feature_columns": best_dataset["feature_columns"],
        "class_name_map": class_name_map,
        "metrics": best_result["metrics"],
        "candidate_results": [_serialise_candidate_result(result) for result in evaluated_candidates],
        "feature_context": feature_context,
        "training_summary": {
            "summary_csv_path": summary_csv_path,
            "events_csv_path": events_csv_path,
            "sample_count": len(best_dataset["y"]),
            "feature_count": len(best_dataset["feature_columns"]),
            "class_counts": dict(Counter(best_dataset["y"].tolist())),
            "training_video_ids": list(best_dataset["video_ids"]),
        },
    }

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(bundle, model_output_path)

    metrics = best_result["metrics"]
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
    print(f"Training accuracy: {metrics['training_accuracy']:.4f}")
    print(f"Model saved to: {model_output_path}")
    print("=" * 60)

    return bundle


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


def feature_row_from_results_dict(results_dict, feature_set="summary"):
    feature_key = str(feature_set).strip().lower()
    if feature_key == "legacy":
        flow_start = _safe_int(results_dict.get("flow_start"), default=0) or 0
        flow_end = _safe_int(results_dict.get("flow_end"), default=0) or 0
        shot_time = max(0, flow_end - flow_start)
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
    if feature_key == "events":
        return feature_rows["events"]
    if feature_key == "combined":
        combined = {}
        combined.update(feature_rows["summary"])
        combined.update(feature_rows["events"])
        return combined

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


def predict_from_feature_row(feature_row, model_path=default_model_path):
    bundle = load_model_bundle(model_path)
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


def predict_from_results_json(results_json_path, model_path=default_model_path):
    bundle = load_model_bundle(model_path)
    bundle_feature_set = bundle.get("feature_set")
    if bundle_feature_set is None:
        feature_set = "legacy"
    else:
        feature_set = bundle_feature_set

    feature_row, raw_results = feature_row_from_results_json(
        results_json_path,
        feature_set=feature_set,
    )
    feature_row = _apply_bundle_feature_context(feature_row, bundle)
    prediction = predict_from_feature_row(feature_row, model_path=model_path)
    prediction["video_name"] = raw_results.get("video_name")
    prediction["results_json"] = results_json_path
    prediction["feature_set"] = feature_set
    prediction["model_type"] = bundle.get("model_type")
    return prediction


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Train and use espresso extraction classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model from exported training CSV files")
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
        "--model-type",
        default=model_type_default,
        choices=list(supported_model_types),
        help="Model family to train, or 'auto' to compare multiple candidates.",
    )
    train_parser.add_argument(
        "--feature-set",
        default=feature_set_default,
        choices=list(supported_feature_sets),
        help="Which exported feature set to use, or 'auto' to compare available sets.",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict extraction class from *_results.json")
    predict_parser.add_argument("--results-json", required=True, help="Path to analysis results JSON")
    predict_parser.add_argument("--model", default=default_model_path, help="Path to trained model")
    predict_parser.add_argument("--print-json", action="store_true", help="Print full JSON output")

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_model(
            summary_csv_path=args.csv,
            events_csv_path=args.events_csv,
            model_output_path=args.model,
            model_type=args.model_type,
            feature_set=args.feature_set,
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
