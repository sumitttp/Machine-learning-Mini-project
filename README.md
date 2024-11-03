Customer Churn Prediction Using Machine Learning
Overview
This project aims to predict customer churn in the telecom sector using machine learning. We analyze the IBM Telco Customer Churn dataset to identify key factors influencing customer retention and develop predictive models for accurate churn classification. The project follows a full machine learning pipeline, from data cleaning to model evaluation, providing valuable insights and potential interventions to reduce churn.

Project Structure
Dataset: IBM Telco Customer Churn dataset
Models Used: Logistic Regression, Random Forest, Support Vector Machine, and a hyperparameter-tuned Random Forest model.
Primary Metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC.
Tools and Libraries: Python, Scikit-Learn, Matplotlib, Pandas, NumPy
Key Files
data/: Contains the IBM Telco Customer Churn dataset.
notebooks/: Jupyter notebooks with step-by-step analysis, feature engineering, model training, and evaluation.
src/: Python scripts for data preprocessing, model training, and evaluation.
README.md: Project overview, setup instructions, and details.
requirements.txt: List of dependencies.
Getting Started
Prerequisites
Python 3.8 or later
Jupyter Notebook
Required libraries from requirements.txt
Installation
Clone the repository:

git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
Install dependencies:

pip install -r requirements.txt
Launch Jupyter Notebook to explore the analysis:

jupyter notebook
Methodology
Data Cleaning and Preprocessing:

Handle missing values, data encoding, normalization, and feature engineering.
Exploratory Data Analysis (EDA):

Identify relationships between features, plot distributions, and visualize churn correlations.
Model Selection:

Trained and evaluated Logistic Regression, Random Forest, and Support Vector Machine models.
Optimized the best-performing model using GridSearchCV for hyperparameter tuning.
Evaluation:

Used cross-validation, confusion matrix, ROC-AUC, and precision/recall metrics to evaluate model performance.
Visualized results with ROC curve and calculated final model performance metrics.
Results
Best Model: Hyperparameter-tuned Random Forest model.
Accuracy: 92.6% on the test set.
Insights:
Key features impacting churn include contract type, monthly charges, and tenure.
Customers with short tenure and higher monthly charges are more likely to churn.
Conclusion
This project demonstrates how machine learning can effectively predict customer churn, helping telecom providers retain customers by identifying high-risk cases. Potential future improvements include exploring additional features or advanced algorithms.

Future Work
Experimenting with deep learning models for enhanced prediction.
Extending the model to real-time churn prediction with streaming data.
Integrating feature selection techniques like PCA to reduce model complexity.
Contributions
Sumit Pandey: Data analysis, model selection, EDA.
Saloni Raut: Data preprocessing, model training, evaluation, hyperparameter tuning.
References
Scikit-Learn Documentation
IBM. “Telecom Customer Churn Data.” IBM Watson Analytics Blog, IBM, 2019.
