# **Obesity Level Classification — Multi-Class Machine Learning Project**

This repository contains a machine learning project focused on predicting **obesity levels** based on eating habits, physical condition, and lifestyle information. The project applies standard data-science workflows including data exploration, preprocessing, model training, and evaluation. Multiple multi-class classification algorithms are implemented and compared.

---

## **Project Overview**

The goal of this project is to build a predictive model that estimates an individual’s **obesity level** using demographic, behavioral, and physical attributes.

The dataset used (“*ObesityDataSet_raw_and_data_sinthetic.csv*”) contains features related to:

* Eating habits
* Physical activity
* Daily lifestyle patterns
* Basic biometric indicators

The target variable is **NObeyesdad**, which includes multiple classes representing different obesity categories.

---

## **Repository Structure**

```
├── Multi-class Classification (3).ipynb   # Main analysis and model training notebook
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
└── README.md
```

---

## **Data Preprocessing**

The notebook includes full preprocessing steps:

* Handling missing values
* Exploring feature distributions
* One-hot encoding of categorical data
* Splitting into train/test sets
* Scaling numerical features using `StandardScaler`

After preprocessing, the dataset is prepared for multi-class modeling.

---

## **Exploratory Data Analysis (EDA)**

The EDA section includes:

* Summary statistics
* Unique value counts
* Pair plots visualized with `seaborn`
* Dataset dimensionality and structure
* Initial inspection of class imbalance and feature correlations

This helps determine which algorithms may perform best.

---

## **Machine Learning Models Implemented**

Three supervised learning approaches were evaluated:

### **1. Logistic Regression (One-vs-Rest — OVR)**

* Uses `LogisticRegression(max_iter=1000)`
* Suitable baseline for multi-class linear decision boundaries

### **2. Logistic Regression (One-vs-One — OVO)**

* Uses `OneVsOneClassifier`
* Creates binary classifiers for each class pair
* Often performs better for multi-class problems

### **3. Decision Tree Classifier**

* `DecisionTreeClassifier(criterion="entropy", max_depth=8)`
* Handles feature interactions effectively
* Less sensitive to scaling
* Achieved strong predictive performance

### **Model Performance Summary**

Both logistic regression (OVR/OVO) and decision tree models performed competitively, with the decision tree being favored due to:

* Simpler interpretability
* Natural handling of non-linear patterns
* Comparable accuracy to logistic regression

---

## **Visualizations**

The notebook provides:

* Pairwise feature plots (colored by obesity level)
* Tree-based classification outputs
* Predicted vs. actual visual interpretations

These visualizations assist in understanding class separation and model behavior.

---

## **Technologies Used**

| Category          | Tools                   |
| ----------------- | ----------------------- |
| Language          | Python                  |
| Data Manipulation | `pandas`, `numpy`       |
| Visualization     | `matplotlib`, `seaborn` |
| Machine Learning  | `scikit-learn`          |

---


## **Future Improvements**

* Hyperparameter tuning (Grid Search / Random Search)
* Model interpretability (SHAP, feature importance)
* Deployment via FastAPI or Streamlit
* Cross-validation to improve performance robustness

