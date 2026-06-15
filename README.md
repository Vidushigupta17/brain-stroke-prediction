# brain-stroke-prediction
📌 Problem Statement

Stroke is one of the leading causes of death and disability worldwide. Early prediction can save lives. This project builds a classification model to predict whether a patient is at risk of a stroke based on medical and lifestyle features.

The core challenge: the dataset is heavily imbalanced — only 783 stroke cases out of 43,400 total samples (less than 2%).


📂 Dataset


Source: Cerebral Stroke Prediction - Imbalanced Dataset (Kaggle)
File: dataset.csv
Shape: 43,400 rows × 12 columns
Target column: stroke (0 = No Stroke, 1 = Stroke)
Class imbalance: 42,617 No Stroke vs 783 Stroke



🔧 Workflow

1. Data Loading & Exploration


Loaded dataset using pandas
Checked shape, data types, null values (bmi had 1,462 missing, smoking_status had 13,292)
Confirmed severe class imbalance in stroke column


2. Preprocessing


Label Encoding of categorical columns using LabelEncoder
Missing value imputation using column means
StandardScaler normalization applied to all features


3. Handling Class Imbalance — Conditional GAN (cGAN)

Instead of simple interpolation (SMOTE), a Conditional GAN was trained to learn the real statistical distribution of stroke patients and generate realistic synthetic minority samples.


Generator: noise + class label → synthetic feature vector
Discriminator: feature vector + class label → real/fake score
Trained for 300 epochs on minority class only
Generated 41,834 synthetic stroke samples to perfectly balance the dataset
Final balanced dataset: 85,234 samples (42,617 each class)


4. Train / Val / Test Split


70% Train (61,580) | 15% Val (10,868) | 15% Test (12,786)
Stratified split to maintain class balance


5. Hybrid Model — CNN + BiLSTM + Transformer

Input (B, 1, F)
     │
 CNN Block (local feature extraction, residual connections)
     │
 BiLSTM Block (2-layer bidirectional, sequential dependencies)
     │
 Transformer Encoder (multi-head self-attention, global context)
     │
 Classifier Head (Linear → GELU → Dropout → Linear)
     │
 Output: stroke probability

Total trainable parameters: 445,585

6. Training with Focal Loss

Focal Loss was used instead of standard BCE to focus the model on hard, rare stroke examples:

FL(p_t) = -α * (1 - p_t)^γ * log(p_t)


alpha = 0.25, gamma = 2.0
Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
Scheduler: CosineAnnealingLR over 50 epochs
Best model saved based on validation F1 score



🏆 Final Results

Metric    Score
Accuracy  98.96%
F1 Score  98.95%
ROC-AUC   99.71%


Train and Val loss converged together (no overfitting)
Val F1 stable at ~0.99 from early epochs
Final Train Loss: 0.0028 | Val Loss: 0.0028

