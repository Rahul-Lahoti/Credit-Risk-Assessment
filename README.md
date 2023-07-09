
[Credit Risk Analysis for Extending Bank Loans Dataset](https://www.kaggle.com/datasets/atulmittal199174/credit-risk-analysis-for-extending-bank-loans)

This project focuses on credit risk assessment using various machine learning models. The goal is to analyze a dataset of bank loans and build models to predict the likelihood of default based on different features.


Data
The project uses a dataset named 'bankloans.csv' containing information about bank loans. The dataset includes various features such as age, education level, employment status, income, debt-to-income ratio, credit and other debts, and the default status.

Code Structure
The code is divided into several sections, each performing a specific task:

The analysis includes
- Data cleaning
- Exploratory data analysis (EDA)
- Handling missing values
- Outlier detection
- Correlation analysis
- Variance inflation factor (VIF) calculation
- Class imbalance analysis


## Analysis Steps

The steps followed in the analysis are as follows:

- Load the dataset and inspect its structure.
- Check for missing values in the dataset.
- Calculate the correlation between different features.
- Calculate the Variance Inflation Factor (VIF) for the features to check for multicollinearity.
- Handle class imbalance in the target variable 'default.'
- Detect and handle outliers in the dataset.
- Split the data into a training set and a test set.
- Fit a Logistic Regression, Random Forest, and Support Vector Machine (SVM) models to the training data.
- Evaluate the models using various metrics, including accuracy, precision, recall, F1 score, and AUC-ROC score.
- Perform feature importance analysis using permutation importance.

## Models Trained

The three models trained on the dataset:

1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. Logistic Regression



```bash
              Model    Accuracy  Precision  Recall    F1 Score    AUC-ROC Score
        Random Forest  0.807143   0.720000  0.473684  0.571429       0.702528
                  SVM  0.864286   0.952381  0.526316  0.677966       0.758256
  Logistic Regression  0.850000   0.814815  0.578947  0.676923       0.764964



