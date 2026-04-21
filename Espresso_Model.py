# Train and run extraction-level classifiers.

import argparse
import csv
import json
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

base_dir = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.join(base_dir, "Analysis")

default_training_csv = os.path.join(analysis_dir, "training_data.csv")
default_model_path = os.path.join(analysis_dir, "extraction_model.joblib")
model_type_default = "random_forest"

default_feature_columns = [
    "shot_time",
    "blonding_rate",
    "channeling_range",
    "channeling_range_norm",
    "channeling_coverage_norm",
]

class_name_map = {
    0: "under",
    1: "ideal",
    2: "over",
}

supported_model_types = ("random_forest", "svm")

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

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


def load_training_data(csv_path, feature_columns=None):
    if feature_columns is None:
        feature_columns = list(default_feature_columns)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data not found: {csv_path}")

    X_rows = []
    y_rows = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = _safe_int(row.get("label"), default=None)
            if label is None:
                continue
            if label not in (0, 1, 2):
                continue

            features = [_safe_float(row.get(col), default=0.0) for col in feature_columns]
            X_rows.append(features)
            y_rows.append(label)

    if not X_rows:
        raise ValueError("No labeled rows found in training CSV.")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int64)
    return X, y, feature_columns


def build_model(model_type="random_forest", random_state=42):
    model_key = str(model_type).strip().lower()
    if model_key not in supported_model_types:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Choose one of: {supported_model_types}"
        )

    if model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample",
        )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
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


def train_model(
    csv_path=default_training_csv,
    model_output_path=default_model_path,
    feature_columns=None,
    random_state=42,
    model_type=model_type_default,
):
    X, y, used_columns = load_training_data(csv_path, feature_columns=feature_columns)
    class_counts = Counter(y.tolist())
    model_key = str(model_type).strip().lower()

    print("\n" + "=" * 60)
    print("TRAINING EXTRACTION LEVEL MODEL")
    print("=" * 60)
    print(f"Dataset: {csv_path}")
    print(f"Samples: {len(y)}")
    print(f"Feature columns: {used_columns}")
    print(f"Class counts: {dict(class_counts)}")
    print(f"Model type: {model_key}")

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    min_class_count = min(class_counts.values())
    use_holdout = len(y) >= 6 and min_class_count >= 2

    model = build_model(model_type=model_key, random_state=random_state)

    metrics = {
        "training_accuracy": None,
        "validation_accuracy": None,
        "classification_report": None,
        "evaluation_mode": "train_only",
    }

    if use_holdout:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=random_state,
            stratify=y,
        )
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        metrics["training_accuracy"] = float(accuracy_score(y_train, y_train_pred))
        metrics["validation_accuracy"] = float(accuracy_score(y_val, y_val_pred))
        metrics["classification_report"] = classification_report(
            y_val,
            y_val_pred,
            labels=[0, 1, 2],
            target_names=["under", "ideal", "over"],
            zero_division=0,
        )
        metrics["evaluation_mode"] = "holdout"
    else:
        model.fit(X, y)
        y_pred = model.predict(X)
        metrics["training_accuracy"] = float(accuracy_score(y, y_pred))
        metrics["evaluation_mode"] = "train_only"

    bundle = {
        "model": model,
        "model_type": model_key,
        "feature_columns": used_columns,
        "class_name_map": class_name_map,
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(bundle, model_output_path)

    print(f"Model saved to: {model_output_path}")
    print(f"Saved model type: {model_key}")
    print(f"Evaluation mode: {metrics['evaluation_mode']}")
    if metrics["training_accuracy"] is not None:
        print(f"Training accuracy: {metrics['training_accuracy']:.4f}")
    if metrics["validation_accuracy"] is not None:
        print(f"Validation accuracy: {metrics['validation_accuracy']:.4f}")
    if metrics["classification_report"]:
        print("\nValidation report:")
        print(metrics["classification_report"])

    return bundle

def load_model_bundle(model_path=default_model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

def feature_row_from_results_dict(results_dict):
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

def feature_row_from_results_json(results_json_path):
    if not os.path.exists(results_json_path):
        raise FileNotFoundError(f"Results JSON not found: {results_json_path}")
    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return feature_row_from_results_dict(data), data

def predict_from_feature_row(feature_row, model_path=default_model_path):
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", list(default_feature_columns))
    names_map = bundle.get("class_name_map", class_name_map)

    x = np.array(
        [[_safe_float(feature_row.get(col), default=0.0) for col in feature_columns]],
        dtype=np.float32,
    )
    predicted_class = int(model.predict(x)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        confidence = float(np.max(proba))

    return {
        "predicted_class": predicted_class,
        "predicted_label": names_map.get(predicted_class, str(predicted_class)),
        "confidence": confidence,
        "feature_vector": {col: _safe_float(feature_row.get(col), default=0.0) for col in feature_columns},
    }

def predict_from_results_json(results_json_path, model_path=default_model_path):
    feature_row, raw_results = feature_row_from_results_json(results_json_path)
    prediction = predict_from_feature_row(feature_row, model_path=model_path)
    prediction["video_name"] = raw_results.get("video_name")
    prediction["results_json"] = results_json_path
    return prediction


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Train and use espresso extraction classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model from training_data.csv")
    train_parser.add_argument("--csv", default=default_training_csv, help="Path to training_data.csv")
    train_parser.add_argument("--model", default=default_model_path, help="Path to save trained model")
    train_parser.add_argument(
        "--model-type",
        default=model_type_default,
        choices=list(supported_model_types),
        help="Model type to train.",
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
            csv_path=args.csv,
            model_output_path=args.model,
            model_type=args.model_type,
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
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Predicted label: {result['predicted_label']}")
            if result["confidence"] is not None:
                print(f"Confidence: {result['confidence']:.4f}")
        return


if __name__ == "__main__":
    main()
