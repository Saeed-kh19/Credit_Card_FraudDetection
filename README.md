# Credit Card Fraud Analysis: Outlier Detection and Distance Measures

A Python-based study of credit card transactions using statistical outlier detection algorithms, manually implemented mathematical distance measures, and data visualization techniques.

This repository contains two separate analytical projects based on the Kaggle Credit Card Fraud Detection dataset:

1. **Multivariate Outlier Detection:** Implementation and comparison of six anomaly detection approaches.
2. **Manual Distance Analysis:** Implementation of Euclidean, Manhattan, and Cosine distances to investigate similarities and differences between legitimate and fraudulent transactions.

Both projects use the same original dataset but have separate implementations, experimental workflows, and objectives.

---

# Dataset

This repository uses the **Credit Card Fraud Detection** dataset published by the Machine Learning Group of Université Libre de Bruxelles (ULB).

**Download the dataset from Kaggle:**

[Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)

## Dataset Installation

Before running either project:

1. Visit the Kaggle dataset page using the link above.
2. Download the dataset.
3. Extract the downloaded archive.
4. Locate the `creditcard.csv` file.
5. Place a copy inside the `data/` directory of the corresponding project.

The dataset is not included in this repository. Users must download it separately from Kaggle.

## Dataset Characteristics

| Property | Description |
|---|---|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Numerical features | 30 |
| Target variable | Class |
| Fraud prevalence | Approximately 0.1727% |

The dataset contains the following numerical features:

- `Time`: Time elapsed since the first transaction.
- `V1`–`V28`: Anonymized features obtained through Principal Component Analysis (PCA).
- `Amount`: Transaction amount.
- `Class`: Transaction label (0 = legitimate, 1 = fraudulent).

The dataset is highly imbalanced, with fraudulent transactions accounting for only a very small proportion of all observations.

---

# Project 1: Multivariate Outlier Detection

## Overview

The first project investigates the detection of anomalous credit card transactions using six statistical and machine learning approaches.

The objective is to apply different outlier detection techniques, compare their detection results, evaluate their performance using ground-truth labels, and identify transactions commonly detected by multiple algorithms.

## Implemented Algorithms

Six outlier detection approaches are implemented using Python machine learning libraries.

| Algorithm | Description |
|---|---|
| Mahalanobis Distance | Identifies observations with unusual multivariate distances from the estimated data distribution. |
| Isolation Forest | Detects anomalies by isolating observations through randomized partitioning. |
| Local Outlier Factor (LOF) | Identifies observations with substantially lower local density than their neighbors. |
| DBSCAN | Identifies noise points that do not belong to sufficiently dense clusters. |
| PCA Reconstruction Error | Detects observations with high reconstruction errors after dimensionality reduction. |
| k-Nearest Neighbors (kNN) Distance | Identifies observations that are unusually distant from their nearest neighbors. |

## Project Workflow

1. **Dataset Loading and Exploration:** Examine the dataset structure, numerical features, missing values, duplicates, and class distribution.
2. **Benchmark Sampling:** Select a stratified sample of 30,000 transactions while preserving the original fraud distribution.
3. **Feature Preprocessing:** Apply logarithmic transformation to transaction amounts and standardize numerical features.
4. **Outlier Detection:** Apply the six detection algorithms to the prepared dataset.
5. **Performance Evaluation:** Compare predicted anomalies with actual fraud labels using classification and ranking metrics.
6. **Algorithm Comparison:** Calculate pairwise intersections and Jaccard Similarity to examine agreement between detection methods.
7. **Common Outlier Identification:** Calculate the strict intersection of all six detection methods and identify transactions detected by at least three methods.
8. **Results Export and Visualization:** Export detection results, evaluation metrics, intersection tables, and visualizations.

## Experimental Setup

The experiment uses a stratified benchmark sample containing:

- 30,000 transactions.
- 29,948 legitimate transactions.
- 52 fraudulent transactions.

Five score-based methods select the 52 highest anomaly scores, using the known fraud count as the alert budget. DBSCAN uses its native noise-point detection instead of this fixed alert count.

The `Class` labels are kept separate from the algorithm input features and are used for evaluation.

## Experimental Results

The following table summarizes the actual experimental results:

| Algorithm | Detected Outliers | True Positives | False Positives | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|---:|
| kNN Distance | 52 | 20 | 32 | 0.3846 | 0.3846 | 0.3846 |
| Mahalanobis Distance | 52 | 16 | 36 | 0.3077 | 0.3077 | 0.3077 |
| Isolation Forest | 52 | 15 | 37 | 0.2885 | 0.2885 | 0.2885 |
| DBSCAN | 309 | 44 | 265 | 0.1424 | 0.8462 | 0.2438 |
| LOF | 52 | 7 | 45 | 0.1346 | 0.1346 | 0.1346 |
| PCA Reconstruction Error | 52 | 0 | 52 | 0.0000 | 0.0000 | 0.0000 |

### Ranking Performance

| Algorithm | ROC-AUC | Average Precision |
|---|---:|---:|
| kNN Distance | 0.9671 | 0.3340 |
| Mahalanobis Distance | 0.9481 | 0.2188 |
| Isolation Forest | 0.9414 | 0.2345 |
| LOF | 0.9251 | 0.0979 |
| PCA Reconstruction Error | 0.5452 | 0.0023 |
| DBSCAN | N/A | N/A |

DBSCAN does not produce a continuous anomaly score in this experiment, so ROC-AUC and Average Precision are not reported for it.

### Key Findings

- **Best F1-Score:** kNN Distance achieved the highest F1-Score (0.3846).
- **Highest Recall:** DBSCAN achieved a Recall of 0.8462, identifying 44 of the 52 fraudulent transactions.
- **Best Ranking Performance:** kNN achieved the highest ROC-AUC (0.9671) and Average Precision (0.3340).
- **Common Outliers:** 11 transactions were identified as outliers by all six algorithms.
- **Consensus Outliers:** 51 transactions were detected by at least three algorithms.

An important observation is that all 11 common outliers were labeled as legitimate transactions in the original dataset. This demonstrates that statistical anomalies are not necessarily fraudulent transactions.

## Common Outlier Analysis

The strict intersection of the six detection methods identified the following transaction IDs:

```text
48529
76679
89606
117639
213160
224569
226690
238412
239532
244865
261843
```

These 11 transactions were detected by every algorithm, but none of them was fraudulent.

Additionally, a consensus-based approach identified 51 transactions detected by at least three methods.

The transaction identifiers represent original dataset row positions rather than actual bank transaction identifiers.

This analysis helps examine algorithm agreement and distinguish consistently detected anomalies from observations identified by only a few methods.

---

# Project 2: Manual Distance Analysis

## Overview

The second project investigates similarities and differences between credit card transactions using manually implemented mathematical distance measures and data visualization techniques.

The main objective is to calculate distances between data points using different mathematical approaches, visualize the resulting distance patterns, and investigate whether legitimate and fraudulent transactions exhibit different characteristics in the feature space.

Unlike the first project, this assignment focuses on implementing distance calculations manually in Python rather than using ready-made distance functions from libraries.

## Manually Implemented Distance Measures

Three mathematical distance measures are implemented in `distance_analysis/src/distance_measures.py`.

No ready-made distance calculation functions from SciPy or Scikit-learn are used.

| Distance Measure | Description |
|---|---|
| Euclidean Distance | Measures the straight-line distance between two points in a multidimensional space. |
| Manhattan Distance | Calculates the sum of absolute differences between corresponding feature values. |
| Cosine Distance | Measures the difference in direction between two feature vectors. |

The mathematical definitions are:

**Euclidean Distance**

\[
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
\]

**Manhattan Distance**

\[
d(x,y)=\sum_{i=1}^{n}|x_i-y_i|
\]

**Cosine Distance**

\[
d(x,y)=1-\frac{x\cdot y}{\|x\|\|y\|}
\]

## Data Preprocessing and Experimental Setup

The original dataset is used for general exploratory visualizations, while a balanced sample is selected for pairwise distance calculations.

The distance-analysis sample contains:

| Property | Value |
|---|---:|
| Total sampled transactions | 200 |
| Legitimate transactions | 100 |
| Fraudulent transactions | 100 |
| Numerical features | 30 |
| Feature matrix shape | 200 × 30 |

The preprocessing operations include:

1. **Feature Selection:** Select the 30 numerical features, excluding the `Class` label.
2. **Amount Transformation:** Apply `log1p` to reduce the skewness of the transaction amount.
3. **Missing Value Handling:** Apply median imputation.
4. **Standardization:** Standardize numerical features using their means and standard deviations.
5. **Balanced Sampling:** Select 100 legitimate and 100 fraudulent transactions using a fixed random state of 42.

The `Class` feature is not included in the numerical distance calculations. It is used only for sampling and categorizing transaction pairs during the analysis.

## Visualizations

The project generates the following charts:

| Visualization | Purpose |
|---|---|
| Class Distribution | Illustrates the imbalance between legitimate and fraudulent transactions. |
| Feature Scatter Plot | Examines the distribution and overlap of legitimate and fraudulent transactions using V14 and V12. |
| Transaction Amount Boxplot | Compares transaction amount distributions between the two categories. |
| Feature Correlation Heatmap | Explores linear relationships between selected numerical features and the encoded class label. |
| Pairwise Distance Heatmaps | Visualizes the Euclidean, Manhattan, and Cosine distance matrices. |
| Distance Group Boxplots | Compares within-class and between-class distance distributions. |

## Pairwise Distance Analysis

Three separate 200 × 200 distance matrices are generated, one for each distance measure.

The pairwise distances are divided into three groups:

| Pair Group | Description | Number of Pairs |
|---|---|---:|
| Legitimate–Legitimate | Distances between two legitimate transactions | 4,950 |
| Fraudulent–Fraudulent | Distances between two fraudulent transactions | 4,950 |
| Legitimate–Fraudulent | Distances between a legitimate and a fraudulent transaction | 10,000 |
| **Total** | | **19,900** |

For every distance measure and pair group, the following statistics are calculated:

- Mean Distance
- Median Distance
- Standard Deviation
- Minimum Distance
- Maximum Distance

## Experimental Results

The following table summarizes the actual pairwise distance results:

| Distance Measure | Pair Group | Mean Distance | Median Distance |
|---|---|---:|---:|
| Euclidean | Legitimate–Legitimate | 7.0283 | 6.5555 |
| Euclidean | Fraudulent–Fraudulent | 23.3565 | 18.4165 |
| Euclidean | Legitimate–Fraudulent | 22.0734 | 16.8138 |
| Manhattan | Legitimate–Legitimate | 29.0608 | 27.5039 |
| Manhattan | Fraudulent–Fraudulent | 90.7505 | 73.4562 |
| Manhattan | Legitimate–Fraudulent | 87.9150 | 68.5779 |
| Cosine | Legitimate–Legitimate | 0.9970 | 1.0215 |
| Cosine | Fraudulent–Fraudulent | 0.4908 | 0.3607 |
| Cosine | Legitimate–Fraudulent | 1.0015 | 1.0104 |

### Within-Class and Between-Class Comparison

The weighted mean within-class distance is calculated by combining the Legitimate–Legitimate and Fraudulent–Fraudulent groups according to their pair counts.

It is then compared with the mean distance between legitimate and fraudulent transactions.

| Distance Measure | Mean Between-Class Distance | Weighted Mean Within-Class Distance |
|---|---:|---:|
| Euclidean | 22.0734 | 15.1924 |
| Manhattan | 87.9150 | 59.9056 |
| Cosine | 1.0015 | 0.7439 |

### Key Findings

- **Euclidean Distance:** Fraudulent–fraudulent pairs have considerably larger average distances than legitimate–legitimate pairs, suggesting greater geometric dispersion among sampled fraudulent transactions.
- **Manhattan Distance:** Shows a similar pattern to Euclidean Distance, confirming the greater dispersion among fraudulent transactions.
- **Cosine Distance:** Fraudulent–fraudulent pairs have substantially smaller average distances, indicating greater directional similarity among their feature vectors.
- **Between-Class Comparison:** All three distance measures produce a higher mean between-class distance than the weighted mean within-class distance.
- **Different Distance Perspectives:** Euclidean and Manhattan distances emphasize geometric dispersion, while Cosine Distance captures directional similarity.

An important detail is that the Fraudulent–Fraudulent mean distances under Euclidean and Manhattan are actually higher than the corresponding Legitimate–Fraudulent mean distances. Therefore, fraudulent transactions are not necessarily close to one another in geometric space.

### Interpretation and Limitations

The findings are based on a balanced sample of 200 transactions and the selected standardized feature space.

The balanced sample does not represent the actual fraud prevalence of the original dataset.

Additionally, the `V1`–`V28` features are anonymized PCA components, so their original real-world meanings cannot be directly interpreted.

The distance comparisons are exploratory and do not establish that any of the three distance measures can independently classify fraudulent transactions.

---

# Technologies and Libraries

The projects are implemented using:

- Python 3.11
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- Jupyter Notebook

The distance-analysis project additionally uses manually implemented Python functions for Euclidean, Manhattan, and Cosine distance calculations.

---

# Repository Structure

```text
Credit_Card_FraudDetection/
│
├── data/
│   └── creditcard.csv
│
├── outputs/
│   └── Outlier Detection results
│
├── distance_analysis/
│   ├── data/
│   │   └── creditcard.csv
│   │
│   ├── outputs/
│   │   ├── dataset_summary.csv
│   │   ├── distance_sample_transactions.csv
│   │   ├── euclidean_distance_matrix.csv
│   │   ├── manhattan_distance_matrix.csv
│   │   ├── cosine_distance_matrix.csv
│   │   ├── distance_group_summary.csv
│   │   ├── class_distribution.png
│   │   ├── feature_scatter.png
│   │   ├── amount_boxplot.png
│   │   ├── feature_correlation_heatmap.png
│   │   ├── distance_heatmaps.png
│   │   └── distance_group_boxplots.png
│   │
│   ├── src/
│   │   └── distance_measures.py
│   │
│   ├── distance_analysis.ipynb
│   └── requirements.txt
│
├── credit_card_fraud_outlier_detection.ipynb
├── run_analysis.py
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The two projects use separate `data/` directories because their notebooks expect the dataset inside their respective project directories. Both dataset copies are excluded from Git.

---

# Installation and Execution

## 1. Clone the Repository

```bash
git clone https://github.com/Saeed-kh19/Credit_Card_FraudDetection.git

cd Credit_Card_FraudDetection
```

## 2. Download the Dataset

Download `creditcard.csv` from Kaggle and place a copy in each of the following directories:

```text
data/creditcard.csv
distance_analysis/data/creditcard.csv
```

## 3. Run Project 1: Outlier Detection

Create and activate the Conda environment:

```bash
conda create -n fraud_outlier_env python=3.11 -y

conda activate fraud_outlier_env
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

`credit_card_fraud_outlier_detection.ipynb`

Execute the notebook cells in order.

The project generates its experimental results and saves them in the root `outputs/` directory.

## 4. Run Project 2: Manual Distance Analysis

Navigate to the distance-analysis directory:

```bash
cd distance_analysis
```

Create and activate its environment:

```bash
conda create -n creditcard_distance python=3.11 -y

conda activate creditcard_distance
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

`distance_analysis.ipynb`

Execute the notebook cells in order.

The generated charts, pairwise distance matrices, and statistical summaries are saved in `distance_analysis/outputs/`.

---

# Output Files

## Outlier Detection

| File | Description |
|---|---|
| `outlier_detection_results.csv` | Detection scores and predictions for benchmark transactions. |
| `outliers_by_method.csv` | Combined outlier records from all algorithms. |
| `evaluation_metrics.csv` | Performance metrics for each detection method. |
| `pairwise_overlap_counts.csv` | Number of common outliers between algorithm pairs. |
| `pairwise_jaccard.csv` | Jaccard Similarity between detection methods. |
| `common_outliers_all_methods.csv` | Transactions detected by all six algorithms. |
| `consensus_3_or_more.csv` | Transactions detected by at least three methods. |
| `results_report.md` | Summary of experimental results and findings. |

## Manual Distance Analysis

| File | Description |
|---|---|
| `dataset_summary.csv` | Dataset characteristics and class distribution. |
| `distance_sample_transactions.csv` | Identifiers and categories of the 200 sampled transactions. |
| `euclidean_distance_matrix.csv` | Pairwise Euclidean distances. |
| `manhattan_distance_matrix.csv` | Pairwise Manhattan distances. |
| `cosine_distance_matrix.csv` | Pairwise Cosine distances. |
| `distance_group_summary.csv` | Descriptive statistics for the three pair categories. |
| `class_distribution.png` | Transaction category distribution. |
| `feature_scatter.png` | Scatter Plot of V14 against V12. |
| `amount_boxplot.png` | Transaction amount distribution by category. |
| `feature_correlation_heatmap.png` | Correlation matrix of selected features. |
| `distance_heatmaps.png` | Visual comparison of the three distance matrices. |
| `distance_group_boxplots.png` | Distance distribution comparison between pair categories. |

---

# Limitations

- The original dataset is highly imbalanced, with fraudulent transactions accounting for approximately 0.1727% of all observations.
- The Outlier Detection experiment uses a 30,000-transaction benchmark sample rather than the entire dataset.
- The detection threshold for five score-based methods is informed by the known fraud count in the benchmark sample.
- DBSCAN is sensitive to its density parameters and feature representation.
- The Manual Distance Analysis experiment uses a balanced sample of 200 transactions for pairwise comparisons.
- The balanced distance-analysis sample does not represent the original fraud prevalence.
- The `V1`–`V28` features are anonymized PCA components and cannot be directly interpreted in terms of original transaction characteristics.
- Distance values depend on feature selection, preprocessing, standardization, and the chosen mathematical measure.
- A detected statistical outlier is not necessarily a fraudulent transaction.

---

# Author

**Saeed Khajeh Khalili**

GitHub: [@Saeed-kh19](https://github.com/Saeed-kh19)

Repository: [Credit_Card_FraudDetection](https://github.com/Saeed-kh19/Credit_Card_FraudDetection)

---

# Dataset Reference

Machine Learning Group – Université Libre de Bruxelles (ULB).

[Credit Card Fraud Detection Dataset – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)
