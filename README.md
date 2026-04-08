❤️ Heart Disease Prediction using Logistic Regression
📌 Project Overview

This project implements a Logistic Regression model to predict whether a person has heart disease based on medical and lifestyle attributes.

The workflow includes:

Data preprocessing
Feature encoding
Model training
Prediction
Performance evaluation

📂 Dataset

The dataset used is:

heart_disease.csv
Key Features:
Age
Gender
Blood Pressure
Cholesterol Level
BMI
Smoking
Diabetes
Exercise Habits
Stress Level
Sleep Hours
Alcohol Consumption
Sugar Consumption
Family Heart Disease
HDL / LDL Cholesterol
Target Variable:
Heart Disease Status
1 → Disease Present
0 → No Disease
⚙️ Technologies Used
Python
Pandas
NumPy
Scikit-learn

🔄 Workflow
1. Data Loading
import pandas as pd
d = pd.read_csv("heart_disease.csv")
2. Data Cleaning
Checked missing values
Removed null values using:
d = d.dropna()
3. Data Preprocessing
Binary Encoding:

Converted categorical values into numerical form:

Yes → 1
No → 0
Gender Encoding:
Male → 1
Female → 0
Ordinal Encoding:

For features like:

Alcohol Consumption
Stress Level
Exercise Habits

Mapping:

High → 2
Medium → 1
Low → 0
4. Feature Selection
x = d[['Age','Gender','Blood Pressure','Cholesterol Level','Exercise Habits',
       'Smoking','Family Heart Disease','Diabetes','BMI','High Blood Pressure',
       'Low HDL Cholesterol','High LDL Cholesterol','Alcohol Consumption',
       'Stress Level','Sleep Hours','Sugar Consumption']]

Target variable:

y = d['Heart Disease Status']
5. Train-Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
6. Model Training
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
7. Prediction
y_pred = lr.predict(x_test)
8. Model Evaluation
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

📊 Output
The model outputs predictions indicating whether a person is likely to have heart disease.
Performance is measured using accuracy score.

📈 Possible Improvements
Use additional metrics:
Precision
Recall
F1-score
Apply feature scaling
Try other models:
Decision Tree
Random Forest
Gradient Boosting
Perform hyperparameter tuning

🚀 How to Run
Install dependencies:
pip install pandas numpy scikit-learn
Place dataset in the same directory
Run the notebook:
jupyter notebook Logisticregression.ipynb

📌 Conclusion
This project demonstrates a basic machine learning pipeline using Logistic Regression for classification. It highlights how preprocessing and feature encoding significantly impact model performance.
