# brain-stroke-prediction
📌 Problem Statement
Stroke is one of the leading causes of death and disability worldwide. Early prediction can save lives. This project builds a classification model to predict whether a patient is at risk of a stroke based on medical and lifestyle features.
The core challenge: the dataset is heavily imbalanced — very few positive stroke cases compared to non-stroke cases.

🔧 Workflow

1. Data Loading & Exploration


Loaded dataset using pandas
Checked shape, data types, null values
Explored class distribution — confirmed severe imbalance in stroke column


2. Preprocessing


Label Encoding of categorical columns using LabelEncoder
Missing value imputation using column means
Feature/Target split: X = all columns except stroke, y = stroke


3. Handling Class Imbalance — SMOTE


Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the classes
Visualized class distribution before and after SMOTE with bar charts
Used PCA to create a 2D scatter plot of the resampled data


4. Train-Test Split & Scaling


80/20 stratified train-test split
Applied StandardScaler to normalize features


5. Model Training & Comparison

Trained and evaluated 5 classification models:

Model                 Accuracy  F1 Score 
Random Forest         95.79%    95.88%
XGBoost               95.68%    95.70%
Decision Tree         94.01%    94.10%
SVM                   87.45%    88.04%
Logistic Regression   82.36%    82.84%

Plotted a bar chart comparing Accuracy and F1 Score across all models.

6. Hyperparameter Tuning


Applied GridSearchCV on XGBoost to find the best combination of:

n_estimators, max_depth, learning_rate, subsample



Evaluated the tuned model's final Accuracy and F1 Score


7. Ensemble Methods

Compared 4 ensemble strategies using the Top 3 models (Random Forest, XGBoost, Decision Tree):

Ensemble MethodAccuracyF1 ScoreHard Voting--Soft Voting--Simple Averaging--Stacking (meta: Logistic Regression)--


🏆 Key Findings


Random Forest was the best single model with ~95.8% accuracy and F1 score
SMOTE was critical — without it, models would have been biased toward the majority class
Ensemble methods were explored to further push performance


