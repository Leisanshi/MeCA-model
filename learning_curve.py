#!/usr/bin/env python3
"""Calculate and draw a fixed-C learning curve using all 985 features."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


LEARNING_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def parse_args():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--expected-features", type=int, default=985)
    parser.add_argument("--output-dir", type=Path, default=base / "results" / "learning_curve")
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=5)
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


def internal_oof_auc(X, y, indices, C, seed):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(indices), np.nan, dtype=float)
    for fold, (train_rel, test_rel) in enumerate(cv.split(X[indices], y[indices]), start=1):
        model = build_model(C, seed + fold)
        model.fit(X[indices[train_rel]], y[indices[train_rel]])
        oof[test_rel] = model.decision_function(X[indices[test_rel]])
    return float(roc_auc_score(y[indices], oof))


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    half_width = 1.96 * sd / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sd, mean - half_width, mean + half_width


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_matrix(args)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    outer_splits = list(outer_cv.split(X, y))

    rows = []
    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits, start=1):
        for fraction in LEARNING_FRACTIONS:
            repeats = 1 if fraction == 1.0 else args.repeats
            for repeat in range(1, repeats + 1):
                if fraction == 1.0:
                    learning_train = outer_train.copy()
                else:
                    learning_train, _ = train_test_split(
                        outer_train,
                        train_size=fraction,
                        stratify=y[outer_train],
                        random_state=args.seed + outer_fold * 10000 + repeat * 100,
                    )
                fraction_code = int(fraction * 10)
                model_seed = (
                    args.seed
                    + 40000
                    + outer_fold * 1000
                    + repeat * 10
                    + fraction_code
                )
                model = build_model(args.C, model_seed)
                model.fit(X[learning_train], y[learning_train])
                training_auc = roc_auc_score(y[learning_train], model.decision_function(X[learning_train]))
                validation_auc = roc_auc_score(y[outer_test], model.decision_function(X[outer_test]))
                oof_seed = (
                    args.seed
                    + 50000
                    + outer_fold * 1000
                    + repeat * 10
                    + fraction_code
                )
                oof_training_auc = internal_oof_auc(
                    X, y, learning_train, args.C, oof_seed
                )
                rows.append({"outer_fold": outer_fold, "training_fraction": fraction, "repeat": repeat, "n_train": len(learning_train), "training_auc": training_auc, "oof_training_auc": oof_training_auc, "validation_auc": validation_auc})
        print("Completed learning-curve analysis in outer fold {}/5".format(outer_fold))

    results = pd.DataFrame(rows)
    summary_rows = []
    for fraction in LEARNING_FRACTIONS:
        subset = results.loc[results["training_fraction"].eq(fraction)]
        row = {"training_fraction": fraction, "mean_n_train": float(subset["n_train"].mean()), "n_runs": len(subset)}
        for source, name in (("training_auc", "training"), ("oof_training_auc", "oof_training"), ("validation_auc", "validation")):
            mean, sd, lower, upper = mean_ci(subset[source].values)
            row["mean_{}_auc".format(name)] = mean
            row["{}_auc_sd".format(name)] = sd
            row["{}_auc_ci95_lower".format(name)] = lower
            row["{}_auc_ci95_upper".format(name)] = upper
        row["train_validation_gap"] = row["mean_training_auc"] - row["mean_validation_auc"]
        row["oof_train_validation_gap"] = row["mean_oof_training_auc"] - row["mean_validation_auc"]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    results.to_csv(str(args.output_dir / "learning_curve_results.tsv"), sep="\t", index=False)
    summary.to_csv(str(args.output_dir / "learning_curve_summary.tsv"), sep="\t", index=False)

    x = summary["mean_n_train"].values
    curves = [
        ("training", "Apparent training AUC", "#2f6f9f", "o"),
        ("oof_training", "Internal OOF AUC", "#3f7d4a", "s"),
        ("validation", "Outer CV validation AUC", "#c54e3d", "o"),
    ]
    plt.figure(figsize=(6.4, 4.8))
    for name, label, color, marker in curves:
        mean = summary["mean_{}_auc".format(name)].values
        lower = summary["{}_auc_ci95_lower".format(name)].values
        upper = summary["{}_auc_ci95_upper".format(name)].values
        plt.plot(x, mean, marker=marker, color=color, label=label)
        plt.fill_between(x, lower, upper, color=color, alpha=0.18)
    plt.xlabel("Number of training samples")
    plt.ylabel("ROC AUC")
    plt.ylim(0.45, 1.02)
    plt.title("Learning curve for the LinearSVC pipeline")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(str(args.output_dir / "learning_curve.png"), dpi=220)
    plt.close()
    print("Fixed C: {}".format(args.C))
    print("Results written to {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
