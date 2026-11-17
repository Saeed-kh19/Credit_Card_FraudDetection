#!/usr/bin/env python3
"""Library-based outlier detection for the Kaggle credit-card fraud dataset.

All detection algorithms are provided by scikit-learn:
  * Mahalanobis distance: sklearn.covariance.LedoitWolf.mahalanobis
  * Isolation Forest: sklearn.ensemble.IsolationForest
  * Local Outlier Factor: sklearn.neighbors.LocalOutlierFactor
  * DBSCAN: sklearn.cluster.DBSCAN
  * PCA reconstruction: sklearn.decomposition.PCA
  * kNN distance: sklearn.neighbors.NearestNeighbors

This project does NOT manually implement the internal mathematics of these
algorithms. NumPy/Pandas are used for data preparation, thresholds, evaluation,
and combining the row IDs detected by the estimators.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Library-based outlier detection on the Kaggle credit-card fraud dataset."
    )
    parser.add_argument(
        "--data", type=Path, default=project_dir / "data" / "creditcard.csv",
        help="Path to creditcard.csv (default: ./data/creditcard.csv).",
    )
    parser.add_argument(
        "--max-rows", type=int, default=30000,
        help="Shared stratified benchmark sample size; use 0 for the full dataset.",
    )
    parser.add_argument(
        "--output", type=Path, default=project_dir / "outputs",
        help="Directory in which result files and plots will be saved.",
    )
    parser.add_argument(
        "--dbscan-eps-quantile", type=float, default=0.98,
        help="Quantile of the DBSCAN k-distance distribution used for eps (0.50 to 1.0).",
    )
    return parser.parse_args()


def load_and_validate(path: Path) -> Tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the project's data/ folder."
        )

    print("[1/8] Loading and validating dataset...")
    df = pd.read_csv(path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from the CSV: {missing}")

    df = df[required_columns].copy()
    # This is a reproducible row key, not a transaction ID supplied by Kaggle.
    df.insert(0, "Transaction_ID", np.arange(len(df), dtype=np.int64))

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Class contains missing or non-numeric values.")
    labels = set(df[TARGET_COLUMN].dropna().unique().tolist())
    if not labels.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Class must contain only 0 and 1; found {sorted(labels)}")

    print(f"  Rows: {len(df):,}")
    print(f"  Numeric input features: {len(FEATURE_COLUMNS)}")
    print(f"  Missing feature cells: {int(df[FEATURE_COLUMNS].isna().sum().sum()):,}")
    print(f"  Duplicate data rows (excluding generated ID): {int(df[required_columns].duplicated().sum()):,}")
    print(f"  Class counts: {df[TARGET_COLUMN].value_counts().sort_index().to_dict()}")
    return df, FEATURE_COLUMNS.copy()


def select_benchmark_sample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    print("[2/8] Selecting a shared benchmark sample...")
    if max_rows <= 0 or max_rows >= len(df):
        sample = df.copy()
        print("  Full dataset selected.")
    else:
        if max_rows < 1000:
            warnings.warn("A sample smaller than 1,000 rows may give unstable evaluation results.")
        if max_rows < 4:
            raise ValueError("max_rows must be at least 4.")
        positions = np.arange(len(df))
        selected, _ = train_test_split(
            positions,
            train_size=max_rows,
            random_state=RANDOM_STATE,
            stratify=df[TARGET_COLUMN].to_numpy(),
        )
        sample = df.iloc[selected].sort_values("Transaction_ID").reset_index(drop=True)
        print(f"  Selected {len(sample):,} rows ({len(sample) / len(df):.1%} of the dataset).")
        print("  Stratification is used only to preserve class proportions in the evaluation sample.")

    n_fraud = int(sample[TARGET_COLUMN].sum())
    print(f"  Sample class counts: {sample[TARGET_COLUMN].value_counts().sort_index().to_dict()}")
    if n_fraud < 10:
        warnings.warn("The sample contains fewer than 10 fraud cases; increase --max-rows for more stable metrics.")
    return sample


def preprocess_features(sample: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    print("[3/8] Preprocessing features...")
    X_df = sample[feature_columns].copy()

    # Dataset-specific transform to reduce the right skew of transaction amounts.
    X_df["Amount"] = np.log1p(X_df["Amount"].clip(lower=0))

    # Standard scikit-learn preprocessing transformers; no detector sees Class.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = np.asarray(X_scaled, dtype=np.float64)

    if not np.isfinite(X_scaled).all():
        raise ValueError("Preprocessing generated NaN or infinite values.")
    print(f"  X shape: {X_scaled.shape}")
    print("  Used: log1p(Amount), SimpleImputer(median), StandardScaler.")
    return X_scaled


def top_k_predictions(scores: np.ndarray, k: int, transaction_ids: np.ndarray) -> np.ndarray:
    """Select the k largest anomaly scores; stable row-ID tie-breaking."""
    n = len(scores)
    k = max(1, min(int(k), n))
    order = np.lexsort((transaction_ids, -np.asarray(scores)))
    predictions = np.zeros(n, dtype=np.int8)
    predictions[order[:k]] = 1
    return predictions


def run_detectors(
    X: np.ndarray,
    transaction_ids: np.ndarray,
    fraud_rate: float,
    dbscan_eps_quantile: float = 0.98,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """Fit six scikit-learn implementations and return scores/predictions.

    `fraud_rate` is used only as a common score-threshold/contamination prior.
    The target vector itself is never passed into this function.
    """
    print("[4/8] Fitting scikit-learn outlier detectors...")
    n_samples, n_features = X.shape
    if n_samples < 3:
        raise ValueError("At least three observations are required.")
    if not (0.0 < fraud_rate < 0.5):
        raise ValueError(f"Expected fraud_rate in (0, 0.5); received {fraud_rate}.")
    if not (0.50 < dbscan_eps_quantile < 1.0):
        raise ValueError("dbscan_eps_quantile must be between 0.50 and 1.0.")

    expected_k = max(1, int(round(n_samples * fraud_rate)))
    scores: Dict[str, np.ndarray] = {}
    predictions: Dict[str, np.ndarray] = {}
    metadata: dict = {"expected_k": expected_k, "contamination_prior": float(fraud_rate)}
    print(f"  Shared alert budget for score-based methods: {expected_k:,} rows.")

    # 1. Mahalanobis Distance: covariance estimation and distance are both
    # performed by scikit-learn's LedoitWolf estimator.
    print("  - Mahalanobis Distance: sklearn.covariance.LedoitWolf")
    covariance_model = LedoitWolf().fit(X)
    mahalanobis_scores = covariance_model.mahalanobis(X)
    scores["Mahalanobis"] = np.asarray(mahalanobis_scores, dtype=float)
    predictions["Mahalanobis"] = top_k_predictions(scores["Mahalanobis"], expected_k, transaction_ids)

    # 2. Isolation Forest: sklearn implements the tree-based detector.
    print("  - Isolation Forest: sklearn.ensemble.IsolationForest")
    isolation_forest = IsolationForest(
        n_estimators=300,
        contamination="auto",
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    isolation_forest.fit(X)
    isolation_scores = -isolation_forest.score_samples(X)  # Larger = more anomalous.
    scores["Isolation Forest"] = isolation_scores
    predictions["Isolation Forest"] = top_k_predictions(isolation_scores, expected_k, transaction_ids)

    # 3. LOF: fit_predict and negative_outlier_factor_ are native LOF APIs.
    print("  - Local Outlier Factor: sklearn.neighbors.LocalOutlierFactor")
    lof_neighbors = min(35, n_samples - 1)
    lof = LocalOutlierFactor(
        n_neighbors=lof_neighbors,
        contamination="auto",
        algorithm="auto",
        n_jobs=-1,
    )
    lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_  # Larger = more anomalous.
    scores["LOF"] = lof_scores
    predictions["LOF"] = top_k_predictions(lof_scores, expected_k, transaction_ids)
    metadata["lof_n_neighbors"] = int(lof_neighbors)

    # Use PCA from scikit-learn to create a lower-dimensional space for DBSCAN
    # and the kNN detector. n_components is capped for small benchmark samples.
    pca_dimensions = min(10, n_features, n_samples - 1)
    density_pca = PCA(n_components=pca_dimensions, svd_solver="full", random_state=RANDOM_STATE)
    X_density = density_pca.fit_transform(X)
    metadata["density_pca_components"] = int(pca_dimensions)
    metadata["density_pca_explained_variance"] = float(density_pca.explained_variance_ratio_.sum())

    # Distances for DBSCAN eps: k-th neighbour distance from the sklearn API.
    min_samples = min(20, n_samples - 1)
    eps_neighbors = NearestNeighbors(n_neighbors=min_samples, algorithm="auto", n_jobs=-1)
    eps_neighbors.fit(X_density)
    eps_distances, _ = eps_neighbors.kneighbors(X_density)
    k_distances_for_eps = eps_distances[:, -1]
    eps = float(np.quantile(k_distances_for_eps, dbscan_eps_quantile))
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"DBSCAN eps calculated as {eps}; cannot continue.")

    # 4. DBSCAN: fit_predict returns cluster IDs; -1 is the library's noise label.
    print("  - DBSCAN: sklearn.cluster.DBSCAN")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    dbscan_labels = dbscan.fit_predict(X_density)
    dbscan_predictions = (dbscan_labels == -1).astype(np.int8)
    predictions["DBSCAN"] = dbscan_predictions
    metadata.update({
        "dbscan_eps": eps,
        "dbscan_eps_quantile": float(dbscan_eps_quantile),
        "dbscan_min_samples": int(min_samples),
        "dbscan_noise_count": int(dbscan_predictions.sum()),
        "dbscan_noise_fraction": float(dbscan_predictions.mean()),
        "dbscan_cluster_count": int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
    })

    # 5. PCA reconstruction error: PCA transform/inverse_transform come from
    # scikit-learn; NumPy only calculates the per-row residual score.
    print("  - PCA Reconstruction Error: sklearn.decomposition.PCA")
    reconstruction_pca = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE)
    X_low_dim = reconstruction_pca.fit_transform(X)
    X_reconstructed = reconstruction_pca.inverse_transform(X_low_dim)
    reconstruction_scores = np.mean(np.square(X - X_reconstructed), axis=1)
    scores["PCA Reconstruction"] = reconstruction_scores
    predictions["PCA Reconstruction"] = top_k_predictions(reconstruction_scores, expected_k, transaction_ids)
    metadata["pca_reconstruction_components"] = int(reconstruction_pca.n_components_)
    metadata["pca_reconstruction_variance"] = float(reconstruction_pca.explained_variance_ratio_.sum())

    # 6. kNN: NearestNeighbors.kneighbors computes the neighbour distances.
    # Query for 21 neighbours because the observation itself is one of them;
    # the last column is therefore the 20th OTHER nearest neighbour.
    print("  - kNN distance: sklearn.neighbors.NearestNeighbors")
    knn_k = min(20, n_samples - 1)
    knn_model = NearestNeighbors(n_neighbors=knn_k + 1, algorithm="auto", n_jobs=-1)
    knn_model.fit(X_density)
    knn_distances, _ = knn_model.kneighbors(X_density)
    knn_scores = knn_distances[:, -1]
    scores["kNN Distance"] = knn_scores
    predictions["kNN Distance"] = top_k_predictions(knn_scores, expected_k, transaction_ids)
    metadata["knn_k"] = int(knn_k)

    metadata["methods"] = list(predictions.keys())
    return scores, predictions, metadata


def calculate_metrics(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    scores: Dict[str, np.ndarray],
) -> pd.DataFrame:
    print("[5/8] Calculating evaluation metrics...")
    rows = []
    for method, pred in predictions.items():
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        row = {
            "Method": method,
            "Detected_Outliers": int(np.sum(pred)),
            "True_Positives": int(tp),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Negatives": int(tn),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred, zero_division=0),
            "F1": f1_score(y_true, pred, zero_division=0),
            "Specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        }
        if method in scores and len(np.unique(y_true)) == 2:
            row["ROC_AUC"] = roc_auc_score(y_true, scores[method])
            row["Average_Precision"] = average_precision_score(y_true, scores[method])
        else:
            row["ROC_AUC"] = np.nan
            row["Average_Precision"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["F1", "Recall"], ascending=False).reset_index(drop=True)


def calculate_overlaps(
    transaction_ids: np.ndarray,
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, set[int], pd.DataFrame]:
    print("[6/8] Comparing detected rows and calculating intersections...")
    method_names = list(predictions)
    detected_sets = {
        name: set(transaction_ids[np.asarray(pred, dtype=bool)].astype(int).tolist())
        for name, pred in predictions.items()
    }

    overlap_counts = pd.DataFrame(index=method_names, columns=method_names, dtype=int)
    jaccard = pd.DataFrame(index=method_names, columns=method_names, dtype=float)
    for method_a in method_names:
        for method_b in method_names:
            shared = detected_sets[method_a] & detected_sets[method_b]
            union = detected_sets[method_a] | detected_sets[method_b]
            overlap_counts.loc[method_a, method_b] = len(shared)
            jaccard.loc[method_a, method_b] = len(shared) / len(union) if union else 0.0

    common_ids = set.intersection(*(detected_sets[name] for name in method_names)) if method_names else set()
    result = pd.DataFrame({"Transaction_ID": transaction_ids.astype(int)})
    if y_true is not None:
        result["Class"] = np.asarray(y_true, dtype=np.int8)
    for method, pred in predictions.items():
        safe_name = method.replace(" ", "_").replace("-", "_")
        result[f"Pred_{safe_name}"] = np.asarray(pred, dtype=np.int8)
    pred_columns = [column for column in result.columns if column.startswith("Pred_")]
    result["Consensus_Count"] = result[pred_columns].sum(axis=1)
    consensus = result[result["Consensus_Count"] >= 3].copy()
    consensus = consensus.sort_values(["Consensus_Count", "Transaction_ID"], ascending=[False, True])
    common = result[result[pred_columns].eq(1).all(axis=1)].copy().sort_values("Transaction_ID")

    overlap_counts.attrs["jaccard"] = jaccard
    return overlap_counts, consensus, common_ids, common


def make_plots(
    y_true: np.ndarray,
    metrics: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    scores: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    print("[7/8] Generating plots...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Class distribution.
    counts = pd.Series(y_true).value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values)
    ax.set_title("Class Distribution in the Benchmark Sample")
    ax.set_ylabel("Number of transactions")
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    # Confusion matrices.
    methods = list(predictions)
    columns = 3
    rows = int(np.ceil(len(methods) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, 4.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, method in zip(axes, methods):
        cm = confusion_matrix(y_true, predictions[method], labels=[0, 1])
        axis.imshow(cm, interpolation="nearest", cmap="Blues")
        axis.set_title(method)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_xticks([0, 1], ["Normal", "Outlier"])
        axis.set_yticks([0, 1], ["Legitimate", "Fraud"])
        for (row, column), value in np.ndenumerate(cm):
            axis.text(column, row, f"{value:,}", ha="center", va="center")
    for axis in axes[len(methods):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    # PR curves for detectors with continuous scores. DBSCAN has only noise labels.
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, score in scores.items():
        precision_values, recall_values, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        ax.plot(recall_values, precision_values, label=f"{method} (AP={ap:.4f})")
    ax.axhline(float(np.mean(y_true)), linestyle="--", label=f"Random baseline={np.mean(y_true):.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves (Continuous Anomaly Scores)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curves.png", dpi=160)
    plt.close(fig)

    # Outlier counts.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ordered = metrics.sort_values("Detected_Outliers", ascending=False)
    ax.bar(ordered["Method"], ordered["Detected_Outliers"])
    ax.set_ylabel("Number flagged as outliers")
    ax.set_title("Outliers Detected by Each Method")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "detected_outlier_counts.png", dpi=160)
    plt.close(fig)


def write_report(
    report_path: Path,
    dataset_rows: int,
    fraud_rate: float,
    sample: pd.DataFrame,
    feature_columns: list[str],
    metrics: pd.DataFrame,
    metadata: dict,
    common: pd.DataFrame,
    consensus: pd.DataFrame,
) -> None:
    class_counts = sample[TARGET_COLUMN].value_counts().to_dict()
    lines = [
        "# Credit Card Fraud Outlier Detection — Results Report",
        "",
        "## 1. Dataset and benchmark",
        "",
        "- Dataset: Kaggle Credit Card Fraud Detection (`mlg-ulb/creditcardfraud`).",
        f"- Original dataset rows: **{dataset_rows:,}**.",
        f"- Numeric input features: **{len(feature_columns)}** (`Time`, `V1`–`V28`, `Amount`).",
        f"- Benchmark rows: **{len(sample):,}**.",
        f"- Benchmark class counts: **{class_counts}** (0 = legitimate, 1 = fraud).",
        f"- Fraud prevalence used as the score threshold prior: **{fraud_rate:.4%}**.",
        "- `Class` is excluded from all model inputs. It is used for stratified benchmark sampling and evaluation only.",
        "- `Transaction_ID` is the zero-based original CSV row position, not a bank-provided transaction ID.",
        "",
        "## 2. Preprocessing",
        "",
        "1. Kept the 30 numeric attributes and separated `Class` from the input matrix.",
        "2. Applied `log1p` to `Amount`.",
        "3. Used `SimpleImputer(strategy='median')` from scikit-learn.",
        "4. Used `StandardScaler` from scikit-learn.",
        "",
        "## 3. Library implementations used",
        "",
        "| Method | Python library implementation | Score / decision |",
        "|---|---|---|",
        "| Mahalanobis Distance | `sklearn.covariance.LedoitWolf` | `mahalanobis(X)`; larger distance is more anomalous. |",
        "| Isolation Forest | `sklearn.ensemble.IsolationForest` | `-score_samples(X)`; larger is more anomalous. |",
        "| LOF | `sklearn.neighbors.LocalOutlierFactor` | `-negative_outlier_factor_`; larger is more anomalous. |",
        "| DBSCAN | `sklearn.cluster.DBSCAN` | `fit_predict(X)`; label `-1` means noise. |",
        "| PCA Reconstruction | `sklearn.decomposition.PCA` | Reconstruction residual after `transform` / `inverse_transform`. |",
        "| kNN Distance | `sklearn.neighbors.NearestNeighbors` | `kneighbors(X)`; distance to the 20th other neighbour. |",
        "",
        "The algorithm internals are provided by scikit-learn. Project code only performs data preparation, score thresholding, evaluation, and set operations.",
        "",
        "## 4. Threshold policy",
        "",
        f"Each continuous-score method flags the **{metadata['expected_k']:,}** largest anomaly scores. This uses the dataset's known fraud prevalence ({fraud_rate:.4%}) as a common alert-budget prior so the score-based methods can be compared at a similar number of alerts. This prior is not passed as a feature and is not used to fit the estimators. DBSCAN uses its native noise labels and can return a different number of outliers.",
        "",
        "## 5. Performance comparison",
        "",
        "Precision, Recall, F1, Specificity, Balanced Accuracy, ROC-AUC, and Average Precision are calculated against `Class`. DBSCAN has no native continuous score, so its ROC-AUC and Average Precision are reported as not applicable.",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 6. DBSCAN configuration",
        "",
        f"- PCA dimensions: **{metadata['density_pca_components']}**.",
        f"- PCA explained variance in the DBSCAN space: **{metadata['density_pca_explained_variance']:.2%}**.",
        f"- `eps`: **{metadata['dbscan_eps']:.6f}** (quantile **{metadata['dbscan_eps_quantile']:.3f}** of the k-distance distribution).",
        f"- `min_samples`: **{metadata['dbscan_min_samples']}**.",
        f"- Clusters excluding noise: **{metadata['dbscan_cluster_count']}**.",
        f"- Noise points: **{metadata['dbscan_noise_count']:,}** ({metadata['dbscan_noise_fraction']:.4%}).",
        "DBSCAN is sensitive to `eps`, `min_samples`, and the PCA representation. The quantile heuristic is a reproducible starting point, not a universally optimal parameter selection.",
        "",
        "## 7. Common outliers and consensus",
        "",
        f"- Rows flagged by all {len(metadata['methods'])} methods: **{len(common):,}**.",
    ]
    if len(common):
        ids = common["Transaction_ID"].astype(int).tolist()
        lines.append("- Common `Transaction_ID` values: " + ", ".join(map(str, ids)))
        lines.append(f"- Fraud cases in the all-method intersection: **{int(common['Class'].sum())} / {len(common)}**.")
    else:
        lines.append("- Common `Transaction_ID` values: **None**. No row was flagged by every method under these settings.")
    lines += [
        f"- Rows flagged by at least three methods: **{len(consensus):,}**.",
        "- See `common_outliers_all_methods.csv`, `consensus_3_or_more.csv`, `pairwise_overlap_counts.csv`, and `pairwise_jaccard.csv`.",
        "",
        "## 8. Limitations and interpretation",
        "",
        "1. Fraudulent transactions and statistical outliers are related but not identical.",
        "2. `V1`–`V28` are anonymized PCA-derived features and do not have directly interpretable business meanings.",
        "3. The default 30,000-row stratified sample reduces computation; results describe that benchmark sample. Set `--max-rows 0` to use all records.",
        "4. The known fraud prevalence is used as a common alert-budget prior for score thresholding. This is appropriate for a controlled comparison but may not be known in a real deployment.",
        "5. DBSCAN is density-based, parameter-sensitive, and does not provide a native continuous anomaly score.",
        "6. The strict all-method intersection may be empty. Pairwise overlaps and the three-or-more-method consensus should also be examined.",
        "",
        "---",
        "Generated automatically by `run_analysis.py`.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def export_results(
    output_dir: Path,
    data_path: Path,
    dataset_rows: int,
    fraud_rate: float,
    sample: pd.DataFrame,
    feature_columns: list[str],
    y: np.ndarray,
    transaction_ids: np.ndarray,
    scores: Dict[str, np.ndarray],
    predictions: Dict[str, np.ndarray],
    metrics: pd.DataFrame,
    overlap_counts: pd.DataFrame,
    consensus: pd.DataFrame,
    common_ids: set[int],
    common: pd.DataFrame,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jaccard = overlap_counts.attrs["jaccard"]

    results = sample[["Transaction_ID", TARGET_COLUMN]].copy().reset_index(drop=True)
    for method, score in scores.items():
        results[f"Score_{method.replace(' ', '_').replace('-', '_')}"] = score
    for method, pred in predictions.items():
        results[f"Pred_{method.replace(' ', '_').replace('-', '_')}"] = pred
    pred_columns = [column for column in results.columns if column.startswith("Pred_")]
    results["Consensus_Count"] = results[pred_columns].sum(axis=1)
    results["Flagged_By_All_Methods"] = results[pred_columns].eq(1).all(axis=1)
    results.to_csv(output_dir / "outlier_detection_results.csv", index=False)

    long_records = []
    for method, pred in predictions.items():
        for position in np.flatnonzero(pred == 1):
            score_value = scores[method][position] if method in scores else np.nan
            long_records.append({
                "Transaction_ID": int(transaction_ids[position]),
                "Method": method,
                "Class": int(y[position]),
                "Anomaly_Score": float(score_value),
            })
    pd.DataFrame(long_records).to_csv(output_dir / "outliers_by_method.csv", index=False)
    metrics.to_csv(output_dir / "evaluation_metrics.csv", index=False)
    overlap_counts.to_csv(output_dir / "pairwise_overlap_counts.csv")
    jaccard.to_csv(output_dir / "pairwise_jaccard.csv")
    common.to_csv(output_dir / "common_outliers_all_methods.csv", index=False)
    consensus.to_csv(output_dir / "consensus_3_or_more.csv", index=False)

    for method, pred in predictions.items():
        selected = sample.loc[pred == 1, ["Transaction_ID", TARGET_COLUMN]].copy()
        if method in scores:
            selected["Anomaly_Score"] = scores[method][pred == 1]
        filename = f"outliers_{method.lower().replace(' ', '_').replace('-', '_')}.csv"
        selected.to_csv(output_dir / filename, index=False)

    make_plots(y, metrics, predictions, scores, output_dir)
    write_report(
        report_path=output_dir / "results_report.md",
        dataset_rows=dataset_rows,
        fraud_rate=fraud_rate,
        sample=sample,
        feature_columns=feature_columns,
        metrics=metrics,
        metadata=metadata,
        common=common,
        consensus=consensus,
    )

    summary = {
        "dataset_path": str(data_path),
        "dataset_rows": int(dataset_rows),
        "sample_rows": int(len(sample)),
        "sample_fraud_rows": int(y.sum()),
        "sample_fraud_rate": float(y.mean()),
        "full_dataset_fraud_rate": float(fraud_rate),
        "transaction_id_definition": "zero-based original CSV row position",
        "methods": list(predictions),
        "all_method_intersection_count": int(len(common_ids)),
        "all_method_intersection_transaction_ids": sorted(map(int, common_ids)),
        **metadata,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df, feature_columns = load_and_validate("/data/creditcard.csv")
    dataset_rows = len(df)
    fraud_rate = float(df[TARGET_COLUMN].mean())
    sample = select_benchmark_sample(df, args.max_rows)

    X = preprocess_features(sample, feature_columns)
    y = sample[TARGET_COLUMN].astype(int).to_numpy()
    transaction_ids = sample["Transaction_ID"].astype(int).to_numpy()

    scores, predictions, metadata = run_detectors(
        X=X,
        transaction_ids=transaction_ids,
        fraud_rate=fraud_rate,
        dbscan_eps_quantile=args.dbscan_eps_quantile,
    )
    metrics = calculate_metrics(y, predictions, scores)
    overlaps, consensus, common_ids, common = calculate_overlaps(transaction_ids, predictions, y_true=y)

    export_results(
        output_dir=args.output,
        data_path=args.data,
        dataset_rows=dataset_rows,
        fraud_rate=fraud_rate,
        sample=sample,
        feature_columns=feature_columns,
        y=y,
        transaction_ids=transaction_ids,
        scores=scores,
        predictions=predictions,
        metrics=metrics,
        overlap_counts=overlaps,
        consensus=consensus,
        common_ids=common_ids,
        common=common,
        metadata=metadata,
    )

    print("[8/8] Finished.")
    display_columns = ["Method", "Detected_Outliers", "True_Positives", "Precision", "Recall", "F1", "Average_Precision"]
    print("\nPerformance summary:")
    print(metrics[display_columns].to_string(index=False))
    print(f"\nOutliers common to all {len(predictions)} methods: {len(common_ids)}")
    print("Common Transaction_ID values:", sorted(common_ids) if common_ids else "None")
    print(f"\nAll output files saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
