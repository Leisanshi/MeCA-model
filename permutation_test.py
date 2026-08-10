#!/usr/bin/env python3
"""Run a label-permutation test for the complete z-score/RFE/LinearSVC pipeline."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def parse_args():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--expected-features", type=int, default=1381)
    parser.add_argument("--output-dir", type=Path, default=base / "results" / "permutation_test")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-select", type=int, default=985)
    parser.add_argument("--rfe-step", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--backend", choices=("loky", "threading"), default="loky")
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


def build_selector(seed, n_select, step):
    estimator = LinearSVC(C=1.0, penalty="l2", loss="squared_hinge", dual=True, tol=1e-4, max_iter=20000, random_state=seed)
    return RFE(estimator=estimator, n_features_to_select=n_select, step=step)


def cross_validated_auc(X, y, splits, n_select, step, seed):
    oof = np.full(len(y), np.nan, dtype=float)
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_index])
        X_test = scaler.transform(X[test_index])
        selector = build_selector(seed + fold, n_select, step)
        selector.fit(X_train, y[train_index])
        oof[test_index] = selector.estimator_.decision_function(selector.transform(X_test))
    if np.isnan(oof).any():
        raise RuntimeError("Not all samples received an out-of-fold score")
    return float(roc_auc_score(y, oof))


def one_permutation(number, X, permuted_y, splits, n_select, step, seed):
    auc = cross_validated_auc(X, permuted_y, splits, n_select, step, seed + number * 10)
    return number, auc


def main():
    args = parse_args()
    if args.n_permutations < 1:
        raise ValueError("n-permutations must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_matrix(args)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits = list(cv.split(X, y))
    observed_auc = cross_validated_auc(X, y, splits, args.n_select, args.rfe_step, args.seed)

    # Permuting labels independently within the five held-out fold blocks keeps
    # class counts fixed while breaking the sample-label association globally.
    rng = np.random.RandomState(args.seed + 5000)
    permuted_labels = []
    for _ in range(args.n_permutations):
        permuted = y.copy()
        for _, test_index in splits:
            permuted[test_index] = rng.permutation(y[test_index])
        permuted_labels.append(permuted)
    values = Parallel(n_jobs=args.n_jobs, backend=args.backend, verbose=10)(
        delayed(one_permutation)(number, X, labels, splits, args.n_select, args.rfe_step, args.seed + 10000)
        for number, labels in enumerate(permuted_labels, start=1)
    )
    values = sorted(values, key=lambda item: item[0])
    results = pd.DataFrame(values, columns=["permutation", "permutation_oof_auc"])
    results["greater_than_or_equal_observed"] = (results["permutation_oof_auc"] >= observed_auc).astype(int)
    exceedances = int(results["greater_than_or_equal_observed"].sum())
    empirical_p = (1.0 + exceedances) / float(args.n_permutations + 1)
    summary = pd.DataFrame([{"observed_oof_auc": observed_auc, "n_permutations": args.n_permutations, "null_mean_auc": results["permutation_oof_auc"].mean(), "null_sd_auc": results["permutation_oof_auc"].std(ddof=1), "exceedances": exceedances, "empirical_one_sided_p": empirical_p}])
    results.to_csv(str(args.output_dir / "permutation_results.tsv"), sep="\t", index=False)
    summary.to_csv(str(args.output_dir / "permutation_summary.tsv"), sep="\t", index=False)

    null_auc = results["permutation_oof_auc"].values
    bins = np.linspace(min(0.4, float(np.min(null_auc)) - 0.02), max(observed_auc + 0.02, float(np.max(null_auc)) + 0.02), 31)
    plt.figure(figsize=(6.4, 4.8))
    plt.hist(null_auc, bins=bins, color="#777777", edgecolor="white")
    plt.axvline(observed_auc, color="#c62828", linewidth=2, label="Observed OOF AUC = {:.3f}".format(observed_auc))
    plt.xlabel("Permuted full-pipeline OOF ROC AUC")
    plt.ylabel("Number of permutations")
    plt.title("Permutation test for the RFE pipeline")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(str(args.output_dir / "permutation_test.png"), dpi=220)
    plt.close()
    print(summary.to_string(index=False))
    print("Results written to {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
