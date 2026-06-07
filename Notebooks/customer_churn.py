import pandas as pd
import numpy as np

df = pd.read_csv("E:Telco-Customer-Churn.csv")

df.info

df.head()

df.sum().isnull()

df.dtypes

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

print(df['Churn'].unique())

df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
df.head()

df['gender'] = df['gender'].replace({'Female':1, 'Male':0})
df.head()

df = df.drop('SeniorCitizen', axis=1)
df.head()

df.select_dtypes(include='object')

df = df.drop('customerID', axis=1)

from sklearn.preprocessing import LabelEncoder

object_cols_names = df.select_dtypes(include='object').columns

le = LabelEncoder()

# Loop through the column names and apply LabelEncoder to the actual Series
for col_name in object_cols_names:
  df[col_name] = le.fit_transform(df[col_name])

df.head()


df = pd.get_dummies(df, columns=['Contract'])


print("DataFrame dtypes before correlation calculation:")
print(df.dtypes)

# We need to convert all remaining categorical 'object' columns to numerical format.
from sklearn.preprocessing import LabelEncoder

object_cols = df.select_dtypes(include='object').columns

if not object_cols.empty:
    print(f"Converting object columns to numeric: {list(object_cols)}")
    le = LabelEncoder()
    for col in object_cols:
        df[col] = le.fit_transform(df[col])
else:
    print("No object columns found. Proceeding with correlation calculation.")

# Now that all relevant columns are numeric, we can calculate the correlation.
print("\nCorrelation with Churn:")
df.corr()['Churn']

df.corr()['Churn']


df = df.drop(['Partner', 'Dependents', 'tenure', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'TotalCharges', 'Contract_1', 'Contract_2'], axis=1)

df.head()

df['Contract_0'] = df['Contract_0'].astype(int)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

customer_ids = X_test.index

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

df['Churn'].value_counts()

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced', random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

model.predict_proba(X_test)

churn_probabilities = model.predict_proba(X_test)[:, 1]
print(churn_probabilities)

results_df = pd.DataFrame({
    'Customer_ID': customer_ids,
    'Churn_Probability': churn_probabilities
})
print(results_df.head())
    
conditions = [
    (results_df['Churn_Probability'] >= 0.80),
    (results_df['Churn_Probability'] >= 0.50) & (results_df['Churn_Probability'] < 0.80),
    (results_df['Churn_Probability'] < 0.50)
]

choices = ['High Risk', 'Medium Risk', 'Low Risk']

results_df['Risk_Tier'] = np.select(conditions, choices, default='Low Risk')

print(results_df['Risk_Tier'].value_counts())

high_risk_customers = results_df[results_df['Risk_Tier'] == 'High Risk'].sort_values(by='Churn_Probability', ascending=False)

print("Top 5 Customers Most Likely to Churn:")
print(high_risk_customers.head())

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Calculate ROC curve metrics
fpr, tpr, thresholds = roc_curve(y_test, churn_probabilities)
auc_score = roc_auc_score(y_test, churn_probabilities)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Model (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Churn Predictions')
plt.legend()
plt.show()

from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, churn_probabilities, n_bins=10)
plt.figure(figsize=(7,7))
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Our Model')
plt.plot([0,1], [0,1], linestyle='--', color='grey', label = 'Perfect Calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('True Proportion of Churn')
plt.title('Calibration Curve (Reliability Diagram)')
plt.legend()
plt.show()

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, churn_probabilities)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision[:-1], label="Precision", color="green")
plt.plot(thresholds, recall[:-1], label="Recall", color="blue")
plt.xlabel("Probability Threshold")
plt.ylabel("Score")
plt.title("Precision vs. Recall Trade-off")
plt.legend()
plt.grid()
plt.show()

import warnings
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np


calibrated_clf = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    calibrated_clf.fit(X_test, y_test)

calibrated_probs = calibrated_clf.predict_proba(X_test)[:, 1]

results_df = pd.DataFrame({
    'Customer_ID': range(len(calibrated_probs)), 
    'Churn_Probability': calibrated_probs 
})
results_df = pd.DataFrame({
    'Churn_Probability': calibrated_probs 
})

results_df['Customer_ID'] = results_df.index 

conditions = [
    (results_df['Churn_Probability'] >= 0.75),
    (results_df['Churn_Probability'] >= 0.45) & (results_df['Churn_Probability'] < 0.75),
    (results_df['Churn_Probability'] < 0.45)
]

choices = ['High Risk', 'Medium Risk', 'Low Risk']
results_df['Risk_Tier'] = np.select(conditions, choices, default='Low Risk')

results_df.to_csv('customer_churn_risk_predictions.csv', index=False)
print("Columns in results_df:", results_df.columns.tolist())
print(results_df.head())



