# Credit Card Fraud Detection Using Outlier Detection Algorithms

A Python-based project for detecting anomalous credit card transactions using multiple statistical and machine learning approaches. This project implements and compares six outlier detection methods and evaluates their ability to identify fraudulent transactions.

## Project Overview

Credit card fraud detection is an important application of anomaly detection in financial systems. However, fraudulent transactions represent only a very small proportion of real-world financial transactions, making their identification challenging.

The objective of this project is to apply different outlier detection techniques to a real-world credit card transaction dataset, compare their detection results, evaluate their performance using ground-truth labels, and identify transactions commonly detected by multiple algorithms.

## Dataset

This project uses the **Credit Card Fraud Detection** dataset published by the Machine Learning Group of Université Libre de Bruxelles (ULB).

**Download the dataset from Kaggle:**

[Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)

### Dataset Installation

Before running the project:

1. Visit the Kaggle dataset page using the link above.
2. Download the dataset.
3. Extract the downloaded archive.
4. Locate the `creditcard.csv` file.
5. Create a directory named `data` in the project's root directory if it does not already exist.
6. Place `creditcard.csv` inside the `data` directory.

The final directory structure should look like this:

```text
Credit_Card_FraudDetection/
│
├── data/
│   └── creditcard.csv
│
├── outputs/
├── credit_card_fraud_outlier_detection.ipynb
├── run_analysis.py
├── requirements.txt
└── README.md
```

**Note:** The dataset is not included in this repository. Users must download it separately from Kaggle.

### Dataset Characteristics

The original dataset contains:

| Property | Description |
|---|---|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Numerical input features | 30 |
| Target variable | Class |
| Fraud rate | Approximately 0.1727% |

The dataset contains 30 numerical input features:

- `Time`: Time elapsed since the first transaction.
- `V1`–`V28`: Anonymized features obtained through Principal Component Analysis (PCA).
- `Amount`: Transaction amount.
- `Class`: Transaction label (0 = legitimate, 1 = fraudulent).

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

The project follows these main stages:

1. **Dataset Loading and Exploration:** Load the dataset and examine its structure, feature types, missing values, duplicates, and class distribution.
2. **Benchmark Sampling:** Select a stratified sample of 30,000 transactions while preserving the original fraud distribution.
3. **Feature Preprocessing:** Apply logarithmic transformation to the transaction amount and standardize numerical features.
4. **Outlier Detection:** Apply the six detection algorithms to the prepared dataset.
5. **Performance Evaluation:** Compare predicted anomalies with actual fraud labels using classification and ranking metrics.
6. **Algorithm Comparison:** Calculate pairwise intersections and Jaccard Similarity to examine agreement between detection methods.
7. **Common Outlier Identification:** Calculate the strict intersection of all six detection methods and identify transactions detected by at least three methods.
8. **Results Export and Visualization:** Export detection results, evaluation metrics, intersection tables, and visualizations.

## Experimental Results

The experiment uses a stratified benchmark sample containing 30,000 transactions, including 52 fraudulent transactions.

The following table summarizes the actual experimental results:

| Algorithm | Detected Outliers | True Positives | False Positives | F1-Score |
|---|---:|---:|---:|---:|
| kNN Distance | 52 | 20 | 32 | 0.3846 |
| Mahalanobis Distance | 52 | 16 | 36 | 0.3077 |
| Isolation Forest | 52 | 15 | 37 | 0.2885 |
| DBSCAN | 309 | 44 | 265 | 0.2438 |
| LOF | 52 | 7 | 45 | 0.1346 |
| PCA Reconstruction Error | 52 | 0 | 52 | 0.0000 |

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

Additionally, a consensus-based approach was used to identify transactions detected by at least three methods.

This analysis provides a comparison of algorithm agreement and helps distinguish consistently detected anomalies from observations identified by only one or two methods.

## Technologies and Libraries

The project is implemented in Python using the following libraries:

- Python 3.11
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/Saeed-kh19/Credit_Card_FraudDetection.git

cd Credit_Card_FraudDetection
```

### 2. Create a Virtual Environment

Using Conda:

```bash
conda create -n fraud_outlier_env python=3.11 -y

conda activate fraud_outlier_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download `creditcard.csv` from the Kaggle link provided above and place it inside the `data/` directory.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `credit_card_fraud_outlier_detection.ipynb` and execute the notebook cells in order.

The project generates its experimental results and saves them in the `outputs/` directory.

## Output Files

The project generates several output files, including:

| File | Description |
|---|---|
| `outlier_detection_results.csv` | Detection scores and predictions for benchmark transactions. |
| `outliers_by_method.csv` | Combined outlier records from all algorithms. |
| `evaluation_metrics.csv` | Performance evaluation metrics for each method. |
| `pairwise_overlap_counts.csv` | Number of common outliers between algorithm pairs. |
| `pairwise_jaccard.csv` | Jaccard Similarity between detection methods. |
| `common_outliers_all_methods.csv` | Transactions detected by all six algorithms. |
| `consensus_3_or_more.csv` | Transactions detected by at least three methods. |
| `results_report.md` | Summary of experimental results and findings. |

Additional CSV files and visualizations are also generated during execution.

## Limitations

- The primary experiment uses a 30,000-transaction benchmark sample rather than the entire dataset.
- The dataset is highly imbalanced, with fraudulent transactions accounting for approximately 0.1727% of all observations.
- The `V1`–`V28` features are anonymized PCA components and cannot be directly interpreted in terms of original transaction characteristics.
- The detection threshold for five score-based methods is informed by the known fraud prevalence.
- DBSCAN is sensitive to its density parameters and feature representation.
- A detected statistical outlier is not necessarily a fraudulent transaction.

## Author

**Saeed Khajeh Khalili**

GitHub: [@Saeed-kh19](https://github.com/Saeed-kh19)

## Dataset Reference

Machine Learning Group – Université Libre de Bruxelles (ULB).

[Credit Card Fraud Detection Dataset – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)
