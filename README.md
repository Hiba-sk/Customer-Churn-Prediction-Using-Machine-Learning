# Customer Churn Prediction

## Overview

This project focuses on predicting customer churn in the telecommunications industry using Machine Learning techniques. The objective is to identify customers who are likely to leave the service and help businesses take proactive retention measures.

The project includes data preprocessing, feature engineering, model training, churn probability estimation, customer risk segmentation, and business intelligence visualization through Power BI.

---

## Dataset

The project uses the Telco Customer Churn dataset, which contains customer demographic information, account details, subscribed services, and churn status.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Power BI

---

## Project Workflow

### 1. Data Preprocessing

- Handled missing values in `TotalCharges`
- Converted categorical variables into numerical format
- Removed unnecessary columns
- Performed feature scaling using StandardScaler

### 2. Exploratory Data Analysis

- Analyzed customer churn patterns
- Examined feature correlations
- Identified important predictors of churn

### 3. Machine Learning Models

The following classification models were implemented and compared:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### 4. Model Evaluation

Models were evaluated using:

- Classification Report
- Confusion Matrix
- ROC Curve
- ROC-AUC Score
- Precision-Recall Curve
- Calibration Curve

### 5. Churn Probability Analysis

Instead of only predicting churn as Yes/No, the model generates churn probabilities using `predict_proba()`.

Customers are categorized into:

| Risk Tier | Churn Probability |
|------------|------------------|
| High Risk | ≥ 80% |
| Medium Risk | 50% – 79% |
| Low Risk | < 50% |

This helps businesses prioritize customer retention strategies.

### 6. Power BI Dashboard

An interactive Power BI dashboard was developed to visualize:

- Customer Churn Distribution
- Risk Tier Breakdown
- High-Risk Customers
- Churn Probability Analysis
- Business Insights

---

## Key Features

- End-to-end Machine Learning Pipeline
- Multiple Classification Models
- Probability-Based Churn Prediction
- Customer Risk Segmentation
- Power BI Dashboard Visualization
- Business-Oriented Insights

---

## Business Impact

This project helps organizations:

- Identify customers likely to churn
- Reduce customer attrition
- Improve retention strategies
- Prioritize high-risk customers
- Support data-driven decision-making

---

## Repository Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── dashboard/
│   └── Churn_Dashboard.pbix
│
├── outputs/
│   ├── churn_predictions.csv
│   ├── roc_curve.png
│   └── calibration_curve.png
│
├── notebooks/
│   └── churn_prediction.ipynb
│
├── README.md
└── requirements.txt
```

---

## Future Improvements

- Hyperparameter Tuning
- Streamlit Deployment
- Real-Time Prediction API
- Automated Model Retraining

---

## References & Acknowledgements

I would like to acknowledge the following resources that helped guide and support this project:

- Dataset: Telco Customer Churn Dataset from Kaggle
- Project Inspiration and Learning Resource:
  - Dataquest – Machine Learning Projects for Beginners to Advanced
    https://www.dataquest.io/blog/machine-learning-projects-for-beginners-to-advanced/

These resources were used for learning, research, and project development purposes.


