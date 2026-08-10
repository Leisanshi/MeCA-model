#!/usr/bin/env python3
"""Run nested-CV LinearSVC C tuning with all 985 features and draw the result."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def parse_args():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--expected-features", type=int, default=985)
    parser.add_argument("--output-dir", type=Path, default=base / "results" / "hyperparameter_tuning")
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def load_matrix(args):
    separator = "," if args.matrix.suffix.lower() == ".csv" else "\t"
    data = pd.read_csv(str(args.matrix), sep=separator)
    if args.label_column not in data.columns:
        raise ValueError("Missing label column: {}".format(args.label_column))
    feature_names = [column for column in data.columns if column != args.label_column]
    if len(feature_names) != args.expected_features:
        raise ValueError("Expected {} features, found {}".format(args.expected_features, len(feature_names)))
    X = data[feature_names].values.astype(float)
    y = data[args.label_column].values.astype(int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("The label column must contain only 0 and 1")
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains missing or non-finite values")
    return X, y


def build_model(C, seed):
    return Pipeline([
        ("zscore", StandardScaler()),
        ("classifier", LinearSVC(C=C, penalty="l2", loss="squared_hinge", dual=True, tol=1e-4, max_iter=20000, random_state=seed)),
    ])


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    half_width = 1.96 * sd / np.sqrt(len(values))
    return mean, sd, mean - half_width, mean + half_width


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_matrix(args)

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    outer_splits = list(outer_cv.split(X, y))
    rows = []
    for outer_fold, (outer_train, _) in enumerate(outer_splits, start=1):
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed + 100 + outer_fold)
        for inner_fold, (train_rel, valid_rel) in enumerate(inner_cv.split(X[outer_train], y[outer_train]), start=1):
            train_index = outer_train[train_rel]
            valid_index = outer_train[valid_rel]
            for C in C_GRID:
                model = build_model(C, args.seed + outer_fold * 100 + inner_fold)
                model.fit(X[train_index], y[train_index])
                score = model.decision_function(X[valid_index])
                rows.append({"outer_fold": outer_fold, "inner_fold": inner_fold, "C": C, "validation_auc": roc_auc_score(y[valid_index], score)})
        print("Completed tuning in outer fold {}/5".format(outer_fold))

    results = pd.DataFrame(rows)
    summary_rows = []
    for C in C_GRID:
        values = results.loc[results["C"].eq(C), "validation_auc"].values
        mean, sd, lower, upper = mean_ci(values)
        summary_rows.append({"C": C, "mean_inner_validation_auc": mean, "sd_inner_validation_auc": sd, "ci95_lower": lower, "ci95_upper": upper, "n_inner_validation_folds": len(values)})
    summary = pd.DataFrame(summary_rows).sort_values("C")
    selected_C = float(summary.sort_values(["mean_inner_validation_auc", "C"], ascending=[False, True]).iloc[0]["C"])
    results.to_csv(str(args.output_dir / "hyperparameter_inner_cv_results.tsv"), sep="\t", index=False)
    summary.to_csv(str(args.output_dir / "hyperparameter_tuning_summary.tsv"), sep="\t", index=False)

    mean = summary["mean_inner_validation_auc"].values
    lower = mean - summary["ci95_lower"].values
    upper = summary["ci95_upper"].values - mean
    plt.figure(figsize=(6.4, 4.8))
    plt.errorbar(summary["C"].values, mean, yerr=np.vstack([lower, upper]), marker="o", color="#2f6f9f", capsize=3)
    plt.axvline(selected_C, color="#c54e3d", linestyle="--", label="Selected C = {}".format(selected_C))
    plt.xscale("log")
    plt.xlabel("LinearSVC regularization parameter C")
    plt.ylabel("Mean inner-validation ROC AUC")
    plt.title("Hyperparameter tuning")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(str(args.output_dir / "hyperparameter_tuning.png"), dpi=220)
    plt.close()
    print("Selected C: {}".format(selected_C))
    print("Results written to {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
