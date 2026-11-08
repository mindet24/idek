import os
import sys
import joblib
import pandas as pd
import sklearn.tree 

try:
    import case_config as cfg
except ImportError as exc:
    raise SystemExit(
        "ERROR: case_config.py must be available next to score_phase_1.py or on PYTHONPATH."
    ) from exc


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

HERE = app_dir()

# =============================================================================
# #TODO: REFEREE CONFIGURATION -- EDIT ONLY THIS BLOCK
# =============================================================================
# Example:
# PHASE1_MODEL_PATH = r"C:\\Users\\Student\\Desktop\\model.joblib"

PHASE1_MODEL_PATH = r"C:\idektep\realmodel5.joblib"

# -----------------------------------------------------------------------------
# UTILITY
# -----------------------------------------------------------------------------

DATA_DIR = os.path.join(HERE, "data")

# =============================================================================

PHASE = 1
UNSEEN_FILENAME = "unseen_1.csv"
ANSWER_FILENAME = "unseen_1_answer.csv"

MISSING_CLASS = -1
FEATURE_COMPARE_RTOL = 1e-10
FEATURE_COMPARE_ATOL = 1e-12


def read_csv(path, what):
    if not os.path.exists(path):
        raise RuntimeError(f"{what} not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df

def normalise_timestamp(series):
    parsed = pd.to_datetime(series.astype(str).str.strip(), errors="coerce")
    return parsed.dt.strftime(cfg.TIMESTAMP_FORMAT)

def check_columns(df, required, what):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{what} is missing columns {missing}. It has: {list(df.columns)}"
        )

def ensure_numeric_features(df, what):
    converted = df.copy()

    for col in cfg.FEATURES:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")

    bad = converted[cfg.FEATURES].isna() & ~df[cfg.FEATURES].isna()
    if bad.any().any():
        locations = []
        for col in cfg.FEATURES:
            n_bad = int(bad[col].sum())
            if n_bad:
                locations.append(f"{col}: {n_bad}")
        raise RuntimeError(
            f"{what} contains non-numeric feature value(s): " + ", ".join(locations)
        )
    return converted

# -----------------------------------------------------------------------------
# HIDDEN UNSEEN and INTERGRITY CHECKS
# -----------------------------------------------------------------------------

def prepare_unseen(path):
    df = read_csv(path, "unseen file")
    check_columns(df, ["timestamp"] + list(cfg.FEATURES), "unseen file")

    df["ts"] = normalise_timestamp(df["timestamp"])

    if df["ts"].isna().any():
        raise RuntimeError(
            "unseen file has %d unreadable timestamp(s)"
            % int(df["ts"].isna().sum())
        )

    if df["ts"].duplicated().any():
        raise RuntimeError("unseen file has duplicate timestamps")

    return ensure_numeric_features(df, "unseen file")

def prepare_answers(path):
    df = read_csv(path, "answer file")
    check_columns(df, cfg.truth_columns(), "answer file")
    check_columns(df, ["timestamp", "y", "y_label"] + list(cfg.FEATURES), "answer file")

    df["ts"] = normalise_timestamp(df["timestamp"])

    if df["ts"].isna().any():
        raise RuntimeError(
            "answer file has %d unreadable timestamp(s)"
            % int(df["ts"].isna().sum())
        )

    if df["ts"].duplicated().any():
        raise RuntimeError("answer file has duplicate timestamps")

    df = ensure_numeric_features(df, "answer file")

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if df["y"].isna().any():
        raise RuntimeError("answer file has non-numeric y values")
    df["y"] = df["y"].astype(int)

    unknown = sorted(set(df["y"]) - set(cfg.class_ids()))
    if unknown:
        raise RuntimeError(f"answer file has invalid class ids: {unknown}")

    actual_labels = df["y_label"].astype(str).str.strip().str.lower()
    expected_labels = df["y"].map(lambda c: cfg.label_of(int(c)).lower())
    bad_label = actual_labels != expected_labels
    if bad_label.any():
        raise RuntimeError(
            "answer file has %d y/y_label mismatch(es)"
            % int(bad_label.sum())
        )
    return df

def validate_unseen_answer_pair(unseen, answers, phase):
    if len(unseen) != len(answers):
        raise RuntimeError(
            f"Phase {phase}: unseen/answer row counts differ: "
            f"{len(unseen)} vs {len(answers)}"
        )

    unseen_ts = set(unseen["ts"])
    answer_ts = set(answers["ts"])

    missing_in_answer = sorted(unseen_ts - answer_ts)
    missing_in_unseen = sorted(answer_ts - unseen_ts)

    if missing_in_answer or missing_in_unseen:
        msg = [f"Phase {phase}: unseen and answer timestamps do not match."]
        if missing_in_answer:
            msg.append(
                f"  {len(missing_in_answer)} unseen timestamp(s) missing from answer"
            )
        if missing_in_unseen:
            msg.append(
                f"  {len(missing_in_unseen)} answer timestamp(s) missing from unseen"
            )
        raise RuntimeError("\n".join(msg))

    left = (
        unseen[["ts"] + list(cfg.FEATURES)]
        .sort_values("ts")
        .reset_index(drop=True)
    )
    right = (
        answers[["ts"] + list(cfg.FEATURES)]
        .sort_values("ts")
        .reset_index(drop=True)
    )

    if not left["ts"].equals(right["ts"]):
        raise RuntimeError(f"Phase {phase}: timestamp order comparison failed")

    for col in cfg.FEATURES:
        a = left[col]
        b = right[col]

        same_nan = a.isna() & b.isna()
        both_numeric = a.notna() & b.notna()
        close = pd.Series(False, index=a.index)
        close.loc[same_nan] = True
        close.loc[both_numeric] = (
            (a.loc[both_numeric] - b.loc[both_numeric]).abs()
            <= FEATURE_COMPARE_ATOL
            + FEATURE_COMPARE_RTOL * b.loc[both_numeric].abs()
        )

        if not bool(close.all()):
            count = int((~close).sum())
            raise RuntimeError(
                f"Phase {phase}: feature '{col}' differs between unseen and "
                f"answer files in {count} row(s). Wrong files may have been paired."
            )

    return True

# -----------------------------------------------------------------------------
# MAPPING AND METRIC
# -----------------------------------------------------------------------------

def join_and_fill(answers, predictions):
    merged = answers.merge(
        predictions[["ts", "y_pred"]],
        on="ts",
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    merged["y_scored"] = merged["y_pred"].fillna(MISSING_CLASS).astype(int)
    merged["matched"] = merged["_merge"] == "both"

    answer_ts = set(answers["ts"])
    extra = int((~predictions["ts"].isin(answer_ts)).sum())
    return merged, extra

def macro_f1(y_true, y_pred):
    y_true = list(y_true)
    y_pred = list(y_pred)
    per_class = []

    for class_id in cfg.class_ids():
        tp = sum(
            1 for t, p in zip(y_true, y_pred)
            if t == class_id and p == class_id
        )
        fp = sum(
            1 for t, p in zip(y_true, y_pred)
            if t != class_id and p == class_id
        )
        fn = sum(
            1 for t, p in zip(y_true, y_pred)
            if t == class_id and p != class_id
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class.append(
            {
                "class_id": class_id,
                "label": cfg.label_of(class_id),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(1 for t in y_true if t == class_id),
            }
        )

    average = sum(row["f1"] for row in per_class) / len(per_class)
    return average, per_class

def accuracy(y_true, y_pred):
    y_true = list(y_true)
    y_pred = list(y_pred)

    if len(y_true) == 0:
        return 0.0

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)

def lazy_baseline(y_true):
    y_true = list(y_true)
    counts = {c: sum(1 for t in y_true if t == c) for c in cfg.class_ids()}
    majority = max(counts, key=counts.get)
    value, _ = macro_f1(y_true, [majority] * len(y_true))
    return value

# -----------------------------------------------------------------------------
# PHASE REQUIREMENT CHECK
# -----------------------------------------------------------------------------

def check_phase_requirements(model_path):
    unseen_path = os.path.join(DATA_DIR, UNSEEN_FILENAME)
    answer_path = os.path.join(DATA_DIR, ANSWER_FILENAME)

    required = [
        ("unseen dataset", unseen_path),
        ("answer dataset", answer_path),
        ("student model", model_path),
    ]

    missing = []
    for label, path in required:
        if not path or not os.path.isfile(path):
            missing.append((label, path))

    if missing:
        lines = [
            f"Phase {PHASE} requirements are incomplete. Scoring was NOT started."
        ]
        for label, path in missing:
            lines.append(f"  MISSING: {label}: {path or '<not provided>'}")

        lines.append("")
        lines.append(
            f"Put {UNSEEN_FILENAME} and {ANSWER_FILENAME} in the data folder next to this scorer, "
            "then run the same command again."
        )
        raise RuntimeError("\n".join(lines))

    return unseen_path, answer_path

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------

def load_competitor_model(model_path):
    if not os.path.exists(model_path):
        raise RuntimeError(f"model file not found: {model_path}")
    return joblib.load(model_path)

def predict_with_model(model, unseen):
    x = unseen[list(cfg.FEATURES)].astype(float).to_numpy()

    try:
        predicted = model.predict(x)
    except Exception as exc:
        raise RuntimeError(
            "model.predict() failed on the hidden unseen feature columns "
            f"{list(cfg.FEATURES)}: {exc}"
        ) from exc

    numeric = pd.to_numeric(pd.Series(predicted), errors="coerce")
    if numeric.isna().any():
        raise RuntimeError("the model returned predictions that are not class ids")

    numeric = numeric.astype(int)
    unknown = sorted(set(numeric) - set(cfg.class_ids()))
    if unknown:
        raise RuntimeError(
            f"the model predicted class ids that do not exist: {unknown}"
        )

    if len(numeric) != len(unseen):
        raise RuntimeError(
            f"model returned {len(numeric)} predictions for {len(unseen)} unseen rows"
        )

    return numeric.to_numpy()

def estimator_name(model):
    try:
        if hasattr(model, "steps") and model.steps:
            return f"{model.__class__.__name__} -> {model.steps[-1][1].__class__.__name__}"
    except Exception:
        pass
    return model.__class__.__name__

# -----------------------------------------------------------------------------
# PHASE SCORE
# -----------------------------------------------------------------------------

def score_phase_1(model_path, unseen_path, answer_path):
    unseen = prepare_unseen(unseen_path)
    answers = prepare_answers(answer_path)
    validate_unseen_answer_pair(unseen, answers, phase=1)

    model = load_competitor_model(model_path)
    y_pred = predict_with_model(model, unseen)

    predictions = pd.DataFrame(
        {
            "ts": unseen["ts"].to_numpy(),
            "y_pred": y_pred,
        }
    )

    merged, extra = join_and_fill(answers, predictions)
    return merged, extra, estimator_name(model)

# -----------------------------------------------------------------------------
# REPORTING
# -----------------------------------------------------------------------------

def print_report(
    submission_path,
    unseen_path,
    answer_path,
    merged,
    extra,
    per_class,
    macro_score,
    baseline_score,
    model_text=None,
):
    total = len(merged)
    matched = int(merged["matched"].sum())
    correct = int((merged["y"] == merged["y_scored"]).sum())
    acc = accuracy(merged["y"], merged["y_scored"])

    print("")
    print("=" * 64)
    print(f"PHASE {PHASE} SCORE")
    print("=" * 64)

    if hasattr(cfg, "CASE_ID"):
        print(f"Case ID:                 {cfg.CASE_ID}")
    if hasattr(cfg, "CASE_TITLE"):
        print(f"Case:                    {cfg.CASE_TITLE}")

    print(f"Model file:              {submission_path}")
    print(f"Estimator:               {model_text}")

    print(f"Hidden unseen file:      {unseen_path}")
    print(f"Hidden answer file:      {answer_path}")
    print("-" * 64)
    print(f"Answer rows:             {total}")
    print(
        "Matched predictions:     %d  (%.1f%% coverage)"
        % (matched, 100.0 * matched / total if total else 0.0)
    )
    print(f"Missing predictions:     {total - matched}  (scored as wrong)")
    print(f"Extra predictions:       {extra}  (ignored)")
    print("-" * 64)
    print(f"Correct rows:            {correct}")
    print(f"Wrong rows:              {total - correct}")
    print(f"Accuracy:                {acc:.4f}")
    print("-" * 64)
    print("Per class:")

    for row in per_class:
        print(
            "  %d %-10s  precision %.4f  recall %.4f  f1 %.4f  n=%d"
            % (
                row["class_id"],
                row["label"],
                row["precision"],
                row["recall"],
                row["f1"],
                row["support"],
            )
        )

    print("-" * 64)
    print(
        "Always-majority baseline: %.4f  (%.4f / %d points)"
        % (
            baseline_score,
            baseline_score * cfg.MAX_SCORE,
            cfg.MAX_SCORE,
        )
    )
    print(f"Macro-F1:                 {macro_score:.4f}")
    print(
        "PHASE %d SCORE:            %.4f / %d"
        % (
            PHASE,
            macro_score * cfg.MAX_SCORE,
            cfg.MAX_SCORE,
        )
    )

    if macro_score <= baseline_score:
        print("WARN: submission does not beat the constant-majority baseline")

    print("=" * 64)

# -----------------------------------------------------------------------------
# Main scoring coordinator
# -----------------------------------------------------------------------------

def score_submission():
    model_path = os.path.abspath(PHASE1_MODEL_PATH)

    unseen_path, answer_path = check_phase_requirements(
        model_path=model_path,
    )

    print(f"Phase {PHASE} requirements: COMPLETE")
    print("Validating hidden unseen/answer pair...")

    merged, extra, model_text = score_phase_1(
        model_path,
        unseen_path,
        answer_path,
    )

    macro_score, per_class = macro_f1(merged["y"], merged["y_scored"])
    baseline_score = lazy_baseline(merged["y"])

    print_report(
        submission_path=model_path,
        unseen_path=unseen_path,
        answer_path=answer_path,
        merged=merged,
        extra=extra,
        per_class=per_class,
        macro_score=macro_score,
        baseline_score=baseline_score,
        model_text=model_text,
    )

    return macro_score * cfg.MAX_SCORE

def ask_path(prompt):
    try:
        return input(prompt).strip().strip('"').strip("'")
    except EOFError:
        return ""

def main():
    global PHASE1_MODEL_PATH

    print("=" * 50)
    print("REFEREE SCORING SYSTEM")
    print("=" * 50)
    print("Phase 1 : Test trained model on unseen_1")
    print("=" * 50)

    if not PHASE1_MODEL_PATH:
        PHASE1_MODEL_PATH = ask_path("Drag the team's model (.joblib) here, then Enter: ")
    if not PHASE1_MODEL_PATH:
        print("ERROR: No file was given.")
        return

    try:
        print("")
        print("Starting Phase 1...")
        print("Model:", PHASE1_MODEL_PATH)

        score_submission()

    except Exception as e:
        print("")
        print("=" * 50)
        print("SCORING FAILED")
        print("=" * 50)
        print("ERROR:", e)

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            input("\nPress Enter to close...")
        except EOFError:
            pass
