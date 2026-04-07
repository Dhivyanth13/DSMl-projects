Project Overview

This project builds a machine learning classification model to predict whether a customer will purchase a car based on demographic and financial attributes.

The model uses a Decision Tree Classifier trained on structured data.

Dataset
The dataset (car_data.csv) contains the following features:

User ID – Unique identifier for each user
Gender – Male/Female (encoded numerically)
Age – Age of the user
Annual Salary – Income of the user
Purchased – Target variable (0 = No, 1 = Yes)

Workflow
1. Data Loading
Dataset is loaded using pandas.read_csv()
2. Data Inspection
Checked for missing values
Used:
.info() → data types and null values
.describe() → statistical summary
3. Data Preprocessing
Label Encoding applied to Gender column:
Converts categorical values into numeric form
4. Feature Selection
Independent variables (X):
User ID
Gender
Age
Annual Salary
Dependent variable (y):
Purchased
5. Train-Test Split
Split ratio: 75% training / 25% testing
Random state: 42 (for reproducibility)
6. Model Training
Algorithm: Decision Tree Classifier
Parameters:
Criterion: gini
Max depth: 3
7. Prediction
Model predicts purchase behavior on test data
8. Evaluation
Metric used: Accuracy Score
9. Visualization
Decision tree is visualized using plot_tree() from sklearn

Model Details
Type: Supervised Learning (Classification)
Algorithm: Decision Tree
Advantage:
Easy to interpret
Handles non-linear relationships

Output
Predicted values (y_pred)
Accuracy score indicating model performance
Visual representation of decision rules (tree structure)

Technologies Used
Python
Pandas
Scikit-learn
Matplotlib

How to Run

Install required libraries:

pip install pandas scikit-learn matplotlib
Place car_data.csv in the same directory

Run the notebook:

jupyter notebook carpurchase.ipynb

Possible Improvements
Remove User ID (not a meaningful predictive feature)
Use feature scaling
Try other models:
Logistic Regression
Random Forest
Perform hyperparameter tuning
Evaluate with additional metrics:
Precision
Recall
F1-score

Conclusion

The model demonstrates how demographic and financial features can be used to predict purchasing behavior using a simple Decision Tree approach. Accuracy can be improved with better feature engineering and model tuning.
