import json
import math
from functools import lru_cache
from pathlib import Path


RULES_PATH = Path(__file__).with_name("espresso_diagnostic_rules.json")


def _safe_float(value, default=0.0):
    try:
        number = float(value)
        if math.isnan(number):
            return float(default)
        return number
    except (TypeError, ValueError):
        return float(default)


def _safe_float_or_none(value):
    try:
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _finite_curve(values):
    output = []
    for value in values or []:
        parsed = _safe_float_or_none(value)
        if parsed is not None:
            output.append(parsed)
    return output


def _mean(values, default=0.0):
    if not values:
        return float(default)
    return float(sum(values) / len(values))


def _max(values, default=0.0):
    if not values:
        return float(default)
    return float(max(values))


def _min(values, default=0.0):
    if not values:
        return float(default)
    return float(min(values))


def _clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, float(value)))


def _norm_progression(curve, threshold):
    values = _finite_curve(curve)
    if not values:
        return None

    total = len(values)
    for index, value in enumerate(values):
        if value >= threshold:
            return index / max(1.0, float(total - 1))
    return None


def _time_at_threshold(curve, threshold, fps):
    values = _finite_curve(curve)
    if not values:
        return None

    fps_safe = max(1e-6, float(fps))
    for index, value in enumerate(values):
        if value >= threshold:
            return float(index / fps_safe)
    return None


@lru_cache(maxsize=1)
def _load_rules():
    with RULES_PATH.open("r", encoding="utf-8") as file_ref:
        config = json.load(file_ref)
    return config.get("flags", [])


def _round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), int(digits))


def _dominant_axis(channel_lr_bias, channel_tb_bias):
    return "left_right" if abs(channel_lr_bias) >= abs(channel_tb_bias) else "top_bottom"


def _build_eval_context(stats, flag_scores):
    return {
        **stats,
        "abs": abs,
        "max": max,
        "min": min,
        "round": round,
        "round4": lambda value: _round_or_none(value, 4),
        "round_or_none": _round_or_none,
        "dominant_axis": _dominant_axis,
        "flag_score": lambda code: float(flag_scores.get(code, 0.0)),
        "flag_score_max": lambda *codes: max((float(flag_scores.get(code, 0.0)) for code in codes), default=0.0),
    }


def _evaluate_expression(expression, stats, flag_scores):
    context = _build_eval_context(stats, flag_scores)
    return eval(expression, {"__builtins__": {}}, context)


def _make_flag(rule, score, stats, flag_scores):
    evidence = {
        key: _evaluate_expression(expression, stats, flag_scores)
        for key, expression in (rule.get("evidence") or {}).items()
    }
    return {
        "code": rule["code"],
        "title": rule["title"],
        "category": rule["category"],
        "description": rule["description"],
        "score": round(_clamp(score), 4),
        "reason": rule["reason"],
        "evidence": evidence,
    }


def compute_diagnostic_statistics(results_dict):
    fps = max(_safe_float(results_dict.get("fps"), 1.0), 1.0)
    flow_start = _safe_int(results_dict.get("flow_start"), 0)
    flow_end = _safe_int(results_dict.get("flow_end"), flow_start)
    shot_time_s = max(0.0, (flow_end - flow_start + 1) / fps)

    flow_detection = results_dict.get("flow_detection") or {}
    spatial = results_dict.get("channeling_spatial_summary") or {}
    temporal = results_dict.get("channeling_temporal_summary") or {}
    quality = results_dict.get("channeling_quality") or {}

    blond_rate = _safe_float(results_dict.get("blond_rate"), 0.0)
    blond_curve = _finite_curve(results_dict.get("blond_score_curve"))
    brightness_curve = _finite_curve(results_dict.get("brightness_curve"))
    channel_norm_curve = _finite_curve(results_dict.get("channeling_norm_curve"))
    channel_counts_curve = _finite_curve(results_dict.get("channeling_counts"))

    quadrant_totals = spatial.get("quadrant_totals") or {}
    quad_keys = ("top_left", "top_right", "bottom_left", "bottom_right")
    quad_values = {key: _safe_float(quadrant_totals.get(key), 0.0) for key in quad_keys}
    total_quad = sum(quad_values.values())
    quad_shares = (
        {key: value / total_quad for key, value in quad_values.items()}
        if total_quad > 0
        else {key: 0.0 for key in quad_keys}
    )

    sorted_quads = sorted(quad_shares.items(), key=lambda item: item[1], reverse=True)
    dominant_quadrant = sorted_quads[0][0] if sorted_quads else None
    dominant_share = sorted_quads[0][1] if sorted_quads else 0.0
    second_share = sorted_quads[1][1] if len(sorted_quads) > 1 else 0.0
    weakest_quadrant = sorted_quads[-1][0] if sorted_quads else None
    weakest_share = sorted_quads[-1][1] if sorted_quads else 0.0

    left_share = quad_shares["top_left"] + quad_shares["bottom_left"]
    right_share = quad_shares["top_right"] + quad_shares["bottom_right"]
    top_share = quad_shares["top_left"] + quad_shares["top_right"]
    bottom_share = quad_shares["bottom_left"] + quad_shares["bottom_right"]

    left_right_bias = _safe_float(spatial.get("global_left_right_asymmetry"), 0.0)
    top_bottom_bias = _safe_float(spatial.get("global_top_bottom_asymmetry"), 0.0)
    side_bias_strength = max(abs(left_right_bias), abs(top_bottom_bias))
    side_dominance_ratio = max(left_share, right_share) / max(
        1e-6,
        min(left_share, right_share) if min(left_share, right_share) > 0 else 1e-6,
    )
    vertical_dominance_ratio = max(top_share, bottom_share) / max(
        1e-6,
        min(top_share, bottom_share) if min(top_share, bottom_share) > 0 else 1e-6,
    )

    channel_mean_norm = _mean(channel_norm_curve, default=0.0)
    channel_peak_norm = _max(channel_norm_curve, default=0.0)
    channel_std_norm = 0.0
    if channel_norm_curve:
        mean_value = channel_mean_norm
        channel_std_norm = float(
            math.sqrt(sum((value - mean_value) ** 2 for value in channel_norm_curve) / len(channel_norm_curve))
        )
    channel_mean_raw = _safe_float(temporal.get("mean_holes"), _mean(channel_counts_curve, default=0.0))
    channel_peak_raw = _safe_float(temporal.get("peak_holes"), _max(channel_counts_curve, default=0.0))
    channel_high_frame_ratio = (
        float(sum(1 for value in channel_norm_curve if value >= 5.0) / len(channel_norm_curve))
        if channel_norm_curve
        else 0.0
    )
    channel_moderate_frame_ratio = (
        float(sum(1 for value in channel_norm_curve if value >= 2.5) / len(channel_norm_curve))
        if channel_norm_curve
        else 0.0
    )

    blond_20_norm = _norm_progression(blond_curve, 0.2)
    blond_60_norm = _norm_progression(blond_curve, 0.6)
    blond_80_norm = _norm_progression(blond_curve, 0.8)
    blond_20_s = _time_at_threshold(blond_curve, 0.2, fps=fps)
    blond_60_s = _time_at_threshold(blond_curve, 0.6, fps=fps)
    blond_80_s = _time_at_threshold(blond_curve, 0.8, fps=fps)
    blonding_change = (_max(blond_curve, default=0.0) - _min(blond_curve, default=0.0)) if blond_curve else 0.0
    brightness_change = (_max(brightness_curve, default=0.0) - _min(brightness_curve, default=0.0)) if brightness_curve else 0.0

    other_quad_shares = [share for key, share in quad_shares.items() if key != dominant_quadrant]
    mean_other_share = _mean(other_quad_shares, default=0.0)
    dominant_to_second_ratio = dominant_share / max(1e-6, second_share if second_share > 0 else 1e-6)
    dominant_to_mean_other_ratio = dominant_share / max(1e-6, mean_other_share if mean_other_share > 0 else 1e-6)

    other_vs_weakest = [share for key, share in quad_shares.items() if key != weakest_quadrant]
    weakest_to_mean_other_ratio = weakest_share / max(
        1e-6,
        _mean(other_vs_weakest, default=0.0) if other_vs_weakest else 1e-6,
    )

    return {
        "shot_time_s": shot_time_s,
        "flow_start_s": float(flow_start / fps),
        "flow_start_confidence": _safe_float(flow_detection.get("start_confidence"), 0.0),
        "flow_end_confidence": _safe_float(flow_detection.get("end_confidence"), 0.0),
        "flow_quality_score": _safe_float(flow_detection.get("quality_score"), 0.0),
        "channel_quality_score": _safe_float(quality.get("quality_score"), 0.0),
        "channel_valid_ratio": _safe_float(quality.get("valid_detection_ratio"), 0.0),
        "channel_mean_raw": channel_mean_raw,
        "channel_peak_raw": channel_peak_raw,
        "channel_mean_norm": channel_mean_norm,
        "channel_peak_norm": channel_peak_norm,
        "channel_std_norm": channel_std_norm,
        "channel_high_frame_ratio": channel_high_frame_ratio,
        "channel_moderate_frame_ratio": channel_moderate_frame_ratio,
        "channel_spatial_entropy": _safe_float(spatial.get("mean_spatial_entropy"), 0.0),
        "channel_lr_bias": left_right_bias,
        "channel_tb_bias": top_bottom_bias,
        "side_bias_strength": side_bias_strength,
        "left_share": left_share,
        "right_share": right_share,
        "top_share": top_share,
        "bottom_share": bottom_share,
        "side_dominance_ratio": side_dominance_ratio,
        "vertical_dominance_ratio": vertical_dominance_ratio,
        "dominant_quadrant": dominant_quadrant,
        "dominant_quadrant_share": dominant_share,
        "dominant_to_second_ratio": dominant_to_second_ratio,
        "dominant_to_mean_other_ratio": dominant_to_mean_other_ratio,
        "weakest_quadrant": weakest_quadrant,
        "weakest_quadrant_share": weakest_share,
        "weakest_to_mean_other_ratio": weakest_to_mean_other_ratio,
        "blond_rate": blond_rate,
        "blond_20_norm": blond_20_norm,
        "blond_60_norm": blond_60_norm,
        "blond_80_norm": blond_80_norm,
        "blond_20_s": blond_20_s,
        "blond_60_s": blond_60_s,
        "blond_80_s": blond_80_s,
        "blonding_change": blonding_change,
        "brightness_change": brightness_change,
    }


def evaluate_rule_based_flags(stats):
    flags = []
    flag_scores = {}

    for rule in _load_rules():
        score = _clamp(_safe_float(_evaluate_expression(rule["score_expr"], stats, flag_scores), 0.0))
        flag_scores[rule["code"]] = score

        if bool(_evaluate_expression(rule["when_expr"], stats, flag_scores)):
            flags.append(_make_flag(rule, score, stats, flag_scores))

    triggered_codes = {flag["code"] for flag in flags}
    return {
        "flags": flags,
        "flag_codes": [flag["code"] for flag in flags],
        "flag_states": {code: int(code in triggered_codes) for code in flag_scores},
        "flag_scores": {code: round(score, 4) for code, score in flag_scores.items()},
        "flag_count": len(flags),
        "max_flag_score": round(max(flag_scores.values()) if flag_scores else 0.0, 4),
    }


def compute_diagnostics(results_dict):
    stats = compute_diagnostic_statistics(results_dict)
    rule_output = evaluate_rule_based_flags(stats)
    return {
        "stats": stats,
        "flags": rule_output["flags"],
        "flag_codes": rule_output["flag_codes"],
        "flag_states": rule_output["flag_states"],
        "flag_scores": rule_output["flag_scores"],
        "flag_count": rule_output["flag_count"],
        "max_flag_score": rule_output["max_flag_score"],
    }


def flatten_diagnostic_features(diagnostics, prefix="diag"):
    diagnostics = diagnostics or {}
    flattened = {
        f"{prefix}_flag_count": _safe_float(diagnostics.get("flag_count"), 0.0),
        f"{prefix}_max_flag_score": _safe_float(diagnostics.get("max_flag_score"), 0.0),
    }

    for key, value in (diagnostics.get("stats") or {}).items():
        if isinstance(value, str):
            continue
        flattened[f"{prefix}_stat_{key}"] = _safe_float_or_none(value)

    for key, value in (diagnostics.get("flag_states") or {}).items():
        flattened[f"{prefix}_flag_{key}"] = _safe_float_or_none(value)

    for key, value in (diagnostics.get("flag_scores") or {}).items():
        flattened[f"{prefix}_score_{key}"] = _safe_float_or_none(value)

    return flattened


def build_combined_assessment(diagnostics, model_prediction=None):
    diagnostics = diagnostics or {}
    flags = diagnostics.get("flags") or []

    assessment = {
        "flag_count": int(diagnostics.get("flag_count") or 0),
        "top_flags": [flag.get("title") for flag in flags[:3]],
    }

    if model_prediction:
        assessment["predicted_class"] = model_prediction.get("predicted_class")
        assessment["predicted_label"] = model_prediction.get("predicted_label")
        assessment["prediction_confidence"] = model_prediction.get("confidence")

    if flags:
        assessment["explanation"] = "; ".join(flag.get("title") for flag in flags[:3])
    elif model_prediction and model_prediction.get("predicted_label"):
        assessment["explanation"] = f"Model prediction: {model_prediction['predicted_label']}"
    else:
        assessment["explanation"] = "No major rule-based issues were triggered."

    return assessment
