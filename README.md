# Customer Churn Prediction — IBM Telco Dataset

An end-to-end introductory machine learning analysis of customer churn using an openly available **fictional telecom customer dataset** published by IBM. This is an independent portfolio project and does not use proprietary employer or client data.

## Business question
Can customer service and account features help identify customers who are more likely to churn?

## Dataset
- **Source:** [IBM Telco Customer Churn CSV](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)
- **Original records:** 7,043; **columns:** 21
- **After cleaning:** 7,032 records, 20 columns (including target)
- **Churn rate after cleaning:** 26.58%
- **Cleaning:** Converted `TotalCharges` to numeric; removed 11 rows with missing charges; excluded `customerID` from features
- **Important:** This is demonstration data for a fictional telecom company, not real operational customer records.

## Approach
1. Data inspection and cleaning with pandas
2. Basic exploratory analysis of churn distribution
3. Stratified 80/20 train/test split (5,625 training, 1,407 test rows)
4. Numerical scaling and one-hot encoding within scikit-learn pipelines
5. Logistic Regression baseline and XGBoost classification
6. Stratified 5-fold cross-validation on the training split
7. Confusion matrix and ROC curve visualization

## Results

### Stratified 5-fold cross-validation (training split)
| Model | ROC-AUC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8461 | 0.5953 | 0.5465 | 0.6541 |
| XGBoost | 0.8491 | 0.5906 | 0.5278 | 0.6712 |

### Previously examined holdout test split
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8038 | 0.6485 | 0.5722 | 0.6080 | 0.8359 |
| XGBoost | 0.7946 | 0.6375 | 0.5267 | 0.5769 | 0.8386 |

The models have similar discrimination. Logistic Regression is a practical, interpretable baseline with slightly higher F1 and recall at the default threshold. XGBoost has slightly higher ROC-AUC. These small differences do not establish statistically significant superiority.

**Evaluation caveat:** The holdout test results were examined before cross-validation-based model preference was formalized. Treat these results as exploratory rather than as an untouched final assessment. Future model or threshold selection should rely on training-only validation before a fresh final holdout evaluation.

## Reproduce
Install dependencies from `requirements.txt` and run `customer_churn_prediction.ipynb` from top to bottom in Jupyter Notebook. The notebook downloads the publicly available CSV from IBM's GitHub repository, so an internet connection is required.

## Limitations and next steps
- Fictional example dataset; generalization to actual customers is unknown
- No causal interpretation of associations
- No temporal or out-of-time validation
- Future work: threshold tuning on validation folds, calibration, and out-of-time evaluation where possible

## Privacy and provenance
This repository uses an independently sourced public sample dataset. It contains no private user-provided, client, or employer data.
