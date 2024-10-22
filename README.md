# Gst Discrepency Detection Model using XGBoost

## 📘 Project Overview

This project focuses on developing a high-performing **binary classification model** using machine learning techniques. The main goal is to classify input data into two classes (0 or 1) by handling missing values, managing skewed data, performing feature selection, and addressing class imbalance. After exploring several classification models, **XGBoost** was selected as the final model due to its superior performance.

## 📝 Problem Statement

The task is to create a machine learning model that can classify rows of input data based on a binary target variable. The challenges include:

- Handling missing values.
- Managing skewed data.
- Selecting meaningful features.
- Addressing class imbalance.
- Choosing the best model and tuning it for high accuracy.

## 🗂️ Data Information

The dataset contains several features and a target column, where each row represents an observation. Missing values, irrelevant features, and class imbalance posed significant challenges, which were addressed through preprocessing and model optimization.

## 📊 Data Preprocessing and Cleaning

### 2.1 Handling Missing Values

- The dataset contained missing values (`NaN`). We replaced missing values with the **median** of each feature to minimize the impact of outliers.
- Columns with more than 50% missing values, specifically columns 9 and 14, were dropped from the dataset to avoid negatively impacting model performance.

### 2.2 Dropping Irrelevant Features

- Categorical features like 'ID' were identified as irrelevant and dropped from the dataset.
- Features with negligible correlation with the target variable (columns 3, 4, 5, 15, and 17) were removed to reduce noise and improve model efficiency.

## 🛠️ Data Preparation for Modeling

### 3.1 Standardization

- The features of both the training and test datasets were standardized to ensure uniform scaling. This step is crucial for models like **SVM** and **Neural Networks**, which are sensitive to the scale of input data.

### 3.2 Test Data Preprocessing

- Consistency was maintained by applying the same preprocessing steps (handling missing values, dropping irrelevant features, and standardization) to the test dataset.

## 🤖 Model Implementation and Selection

### 4.1 Model Exploration

Several machine learning models were implemented to find the best fit for the binary classification task:

- **Logistic Regression:** Used as a baseline for comparison.
- **Random Forest:** An ensemble method that uses multiple decision trees to improve accuracy and reduce overfitting.
- **Gradient Boosting:** Builds models sequentially, where each model corrects the errors of its predecessor.
- **XGBoost:** An optimized version of Gradient Boosting known for its speed and performance.
- **Artificial Neural Networks (ANN):** Captures complex, non-linear relationships in the data.

### 4.2 Model Evaluation

- Each model was evaluated based on metrics such as **accuracy**, **precision**, **recall**, and **F1-score**. The **XGBoost** model outperformed the others, delivering the highest accuracy.

## 🔧 Hyperparameter Tuning & Cross-Validation

### 5.1 Hyperparameter Tuning

- **Grid Search** and **Random Search** techniques were used to tune hyperparameters like **learning rate**, **max depth**, and **number of estimators** for the XGBoost model. This resulted in a more accurate and robust model.
- Cross-validation (with `cv=3`) was used to ensure the model generalized well to unseen data.

### 5.2 Handling Class Imbalance

- The dataset exhibited **class imbalance** between target labels (1 and 0). Techniques like calculating the **imbalance ratio** and adjusting **class weights** were employed to ensure the model was not biased towards the majority class.

## 🏆 Final Model Performance

### 6.1 Accuracy Scores

- The final **XGBoost model** achieved an accuracy of **94-95%** across individual classes, with an overall accuracy of **95%**.

### 6.2 Confusion Matrix & AUC-ROC Curve

- A **Confusion Matrix** was plotted to visualize the model's performance and misclassification areas.
- The **AUC-ROC curve** was also plotted, resulting in an **AUC score of 0.99**, indicating excellent discriminatory power.

## 📈 Conclusion

This project successfully developed a high-performing binary classification model. Through meticulous **data preprocessing**, **feature selection**, **model tuning**, and **class imbalance handling**, the **XGBoost model** was fine-tuned to deliver high accuracy (95%) and excellent performance as measured by the **AUC-ROC curve**. This model is well-suited for deployment in real-world binary classification tasks.

---

## 🛠️ Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/binary-classification-xgboost.git
