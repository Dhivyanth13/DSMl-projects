Mall Customer Segmentation using K-Means
Overview

This project applies K-Means clustering to segment mall customers based on their purchasing behavior. The segmentation helps identify distinct customer groups using:

Annual Income
Spending Score

The goal is to support targeted marketing strategies and customer analysis.

Dataset

The dataset used is:  Mall_Customers.csv

Features used:
Annual Income (k$)
Spending Score (1-100)

Workflow
1. Data Loading
The dataset is loaded using pandas.
import pandas as pd
d = pd.read_csv("Mall_Customers.csv")
2. Data Inspection
Checked for missing values:
d.isna().sum()
Checked dataset structure:
d.info()
3. Feature Selection
Only relevant features are selected for clustering:

x = d[["Annual Income (k$)", "Spending Score (1-100)"]]
4. Data Scaling
Standardization is applied using StandardScaler to normalize feature values.
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
df = s.fit_transform(x)
5. Model Training (K-Means)
K-Means clustering is applied with 5 clusters.
from sklearn.cluster import KMeans
sd = KMeans(n_clusters=5, random_state=42)
d["cluster"] = sd.fit_predict(df)
6. Visualization
Scatter plot is used to visualize clusters.
import matplotlib.pyplot as plt

plt.scatter(
    d["Annual Income (k$)"],
    d["Spending Score (1-100)"],
    c=d["cluster"]
)
plt.xlabel("income")
plt.ylabel("spending_score")
plt.show()

Output
Each customer is assigned a cluster label (0–4).
Visualization shows how customers are grouped based on income and spending behavior.
Dependencies

Install required libraries:

pip install pandas scikit-learn matplotlib
Key Concepts Used
Unsupervised Learning
K-Means Clustering
Feature Scaling (Standardization)
Data Visualization
Assumptions
Number of clusters is fixed at 5 (not optimized using methods like Elbow Method).
Only two features are used for clustering.
Possible Improvements
Use Elbow Method to find optimal number of clusters.
Include more features (e.g., Age, Gender).
Use advanced clustering (DBSCAN, Hierarchical).
Add cluster interpretation (e.g., high-income/high-spending group). 

Project Structure
├── Mall_Customers.csv
├── KMeans_Mallcustomersegmentation.ipynb
└── README.md
