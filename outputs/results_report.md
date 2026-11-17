# Credit Card Fraud Outlier Detection — Results Report

## 1. Dataset and benchmark

- Dataset: Kaggle Credit Card Fraud Detection (`mlg-ulb/creditcardfraud`).
- Original dataset rows: **284,807**.
- Numeric input features: **30** (`Time`, `V1`–`V28`, `Amount`).
- Benchmark rows: **30,000**.
- Benchmark class counts: **{0: 29948, 1: 52}** (0 = legitimate, 1 = fraud).
- Fraud prevalence used as the score threshold prior: **0.1727%**.
- `Class` is excluded from all model inputs. It is used for stratified benchmark sampling and evaluation only.
- `Transaction_ID` is the zero-based original CSV row position, not a bank-provided transaction ID.

## 2. Preprocessing

1. Kept the 30 numeric attributes and separated `Class` from the input matrix.
2. Applied `log1p` to `Amount`.
3. Used `SimpleImputer(strategy='median')` from scikit-learn.
4. Used `StandardScaler` from scikit-learn.

## 3. Library implementations used

| Method | Python library implementation | Score / decision |
|---|---|---|
| Mahalanobis Distance | `sklearn.covariance.LedoitWolf` | `mahalanobis(X)`; larger distance is more anomalous. |
| Isolation Forest | `sklearn.ensemble.IsolationForest` | `-score_samples(X)`; larger is more anomalous. |
| LOF | `sklearn.neighbors.LocalOutlierFactor` | `-negative_outlier_factor_`; larger is more anomalous. |
| DBSCAN | `sklearn.cluster.DBSCAN` | `fit_predict(X)`; label `-1` means noise. |
| PCA Reconstruction | `sklearn.decomposition.PCA` | Reconstruction residual after `transform` / `inverse_transform`. |
| kNN Distance | `sklearn.neighbors.NearestNeighbors` | `kneighbors(X)`; distance to the 20th other neighbour. |

The algorithm internals are provided by scikit-learn. Project code only performs data preparation, score thresholding, evaluation, and set operations.

## 4. Threshold policy

Each continuous-score method flags the **52** largest anomaly scores. This uses the dataset's known fraud prevalence (0.1727%) as a common alert-budget prior so the score-based methods can be compared at a similar number of alerts. This prior is not passed as a feature and is not used to fit the estimators. DBSCAN uses its native noise labels and can return a different number of outliers.

## 5. Performance comparison

Precision, Recall, F1, Specificity, Balanced Accuracy, ROC-AUC, and Average Precision are calculated against `Class`. DBSCAN has no native continuous score, so its ROC-AUC and Average Precision are reported as not applicable.

| Method             |   Detected_Outliers |   True_Positives |   False_Positives |   False_Negatives |   True_Negatives |   Precision |   Recall |     F1 |   Specificity |   Balanced_Accuracy |   ROC_AUC |   Average_Precision |
|:-------------------|--------------------:|-----------------:|------------------:|------------------:|-----------------:|------------:|---------:|-------:|--------------:|--------------------:|----------:|--------------------:|
| kNN Distance       |                  52 |               20 |                32 |                32 |            29916 |      0.3846 |   0.3846 | 0.3846 |        0.9989 |              0.6918 |    0.9671 |              0.3340 |
| Mahalanobis        |                  52 |               16 |                36 |                36 |            29912 |      0.3077 |   0.3077 | 0.3077 |        0.9988 |              0.6532 |    0.9481 |              0.2188 |
| Isolation Forest   |                  52 |               15 |                37 |                37 |            29911 |      0.2885 |   0.2885 | 0.2885 |        0.9988 |              0.6436 |    0.9414 |              0.2345 |
| DBSCAN             |                 309 |               44 |               265 |                 8 |            29683 |      0.1424 |   0.8462 | 0.2438 |        0.9912 |              0.9187 |  nan      |            nan      |
| LOF                |                  52 |                7 |                45 |                45 |            29903 |      0.1346 |   0.1346 | 0.1346 |        0.9985 |              0.5666 |    0.9251 |              0.0979 |
| PCA Reconstruction |                  52 |                0 |                52 |                52 |            29896 |      0.0000 |   0.0000 | 0.0000 |        0.9983 |              0.4991 |    0.5452 |              0.0023 |

## 6. DBSCAN configuration

- PCA dimensions: **10**.
- PCA explained variance in the DBSCAN space: **39.75%**.
- `eps`: **3.681932** (quantile **0.980** of the k-distance distribution).
- `min_samples`: **20**.
- Clusters excluding noise: **1**.
- Noise points: **309** (1.0300%).
DBSCAN is sensitive to `eps`, `min_samples`, and the PCA representation. The quantile heuristic is a reproducible starting point, not a universally optimal parameter selection.

## 7. Common outliers and consensus

- Rows flagged by all 6 methods: **11**.
- Common `Transaction_ID` values: 48529, 76679, 89606, 117639, 213160, 224569, 226690, 238412, 239532, 244865, 261843
- Fraud cases in the all-method intersection: **0 / 11**.
- Rows flagged by at least three methods: **51**.
- See `common_outliers_all_methods.csv`, `consensus_3_or_more.csv`, `pairwise_overlap_counts.csv`, and `pairwise_jaccard.csv`.

## 8. Limitations and interpretation

1. Fraudulent transactions and statistical outliers are related but not identical.
2. `V1`–`V28` are anonymized PCA-derived features and do not have directly interpretable business meanings.
3. The default 30,000-row stratified sample reduces computation; results describe that benchmark sample. Set `--max-rows 0` to use all records.
4. The known fraud prevalence is used as a common alert-budget prior for score thresholding. This is appropriate for a controlled comparison but may not be known in a real deployment.
5. DBSCAN is density-based, parameter-sensitive, and does not provide a native continuous anomaly score.
6. The strict all-method intersection may be empty. Pairwise overlaps and the three-or-more-method consensus should also be examined.

---
Generated automatically by `run_analysis.py`.