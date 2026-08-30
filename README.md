# 🩺 Breast Cancer Classification --- Machine Learning & Streamlit

An end-to-end machine learning project for classifying breast tumors as
**Benign (B)** or **Malignant (M)** using 30 diagnostic features.

The project explores the effect of **feature standardization, PCA, and
hyperparameter tuning** on Logistic Regression and K-Nearest Neighbors
(KNN), followed by deployment of the final model using Streamlit.

## 🚀 Live Demo
https://breast-cancer-app-o26uhfvbdzyftwa8grmtkb.streamlit.app/

------------------------------------------------------------------------

## 📌 Project Overview

The goal of this project is to build and evaluate classification models
for breast cancer prediction and understand how different preprocessing
and model-selection techniques affect performance.

The workflow includes:

-   Train-test splitting
-   Feature standardization using `StandardScaler`
-   Logistic Regression
-   K-Nearest Neighbors (KNN)
-   PCA for dimensionality reduction
-   Hyperparameter tuning using `GridSearchCV`
-   Accuracy, precision, recall and F1-score evaluation
-   Confusion matrix analysis
-   Saving the final model as a reusable pipeline
-   Deployment using Streamlit

------------------------------------------------------------------------

## 🧠 Machine Learning Workflow

``` text
Dataset
   ↓
Train / Test Split
   ↓
Standardization
   ↓
Baseline Models
   ↓
PCA (90% variance)
   ↓
Model Comparison
   ↓
GridSearchCV
   ↓
Final Model / Pipeline
   ↓
Streamlit Deployment
```

------------------------------------------------------------------------

## 📊 Model Performance

The models were evaluated at different stages to understand the impact
of preprocessing and hyperparameter tuning.

### 1. Before PCA --- After Standardization

After applying `StandardScaler`, the models were evaluated before
dimensionality reduction.

  Model                      Best Setting     Accuracy
  --------------------- ----------------- ------------
  Logistic Regression             Default   **97.37%**
  KNN                     n_neighbors = 9   **96.49%**

This established the standardized-data baseline before applying PCA.

------------------------------------------------------------------------

### 2. After PCA

PCA was applied while retaining approximately **90% of the original
variance**.

-   Original features: **30**
-   PCA components: **7**
-   Variance preserved: **90.87%**

  Model                      Best Setting     Accuracy
  --------------------- ----------------- ------------
  Logistic Regression             Default   **98.25%**
  KNN                     n_neighbors = 7   **96.49%**

The Logistic Regression model improved from approximately **97.37% to
98.25% accuracy** after PCA.

------------------------------------------------------------------------

### 3. After PCA + GridSearchCV

`GridSearchCV` with 5-fold cross-validation was used to search for the
best hyperparameters.

#### KNN

Best parameter found:

``` text
n_neighbors = 1
```

Test-set accuracy:

``` text
94.74%
```

#### Logistic Regression

The tuned Logistic Regression model achieved:

``` text
98.25% test accuracy
```

The final Logistic Regression model maintained the strong performance
observed after PCA.

------------------------------------------------------------------------

## 🏆 Final Model

The final deployed model uses the following workflow:

``` text
StandardScaler
      ↓
PCA
      ↓
Logistic Regression
```

The final Logistic Regression model achieved approximately:

### **98.25% Test Accuracy**

On the test set, the classification report showed:

  Class             Precision     Recall   F1-score
  --------------- ----------- ---------- ----------
  Benign (B)             0.99       0.99       0.99
  Malignant (M)          0.98       0.98       0.98
  **Overall**        **0.98**   **0.98**   **0.98**

Test set size: **114 samples**

The model achieved strong and balanced performance across both classes,
including the malignant class.

> **Important:** Accuracy alone is not sufficient for evaluating a
> medical classification model, so precision, recall, F1-score and other
> evaluation methods were also considered.

------------------------------------------------------------------------

## 📉 PCA --- Principal Component Analysis

The dataset originally contained **30 features**.

PCA reduced these 30 features to **7 principal components** while
preserving approximately **90.87% of the variance**.

This helped reduce the dimensionality of the feature space while
retaining most of the information present in the original data.

------------------------------------------------------------------------

## ⚙️ Hyperparameter Tuning

Rather than manually selecting hyperparameters, `GridSearchCV` was used
with **5-fold cross-validation**.

For KNN, the search evaluated:

``` text
n_neighbors = 1 to 20
```

The best KNN parameter found after PCA was:

``` text
n_neighbors = 1
```

For Logistic Regression, hyperparameters including the regularization
configuration were explored using grid search.

------------------------------------------------------------------------

## 📈 Evaluation Metrics

The project uses several classification metrics:

-   **Accuracy** --- overall percentage of correct predictions.
-   **Precision** --- how many predicted samples of a class were
    actually that class.
-   **Recall** --- how many actual samples of a class were correctly
    identified.
-   **F1-score** --- harmonic mean of precision and recall.
-   **Confusion Matrix** --- shows correct and incorrect predictions for
    each class.
    
------------------------------------------------------------------------

## 🌐 Streamlit Application

The trained pipeline is deployed as an interactive Streamlit web
application.

The application allows users to:

1.  Enter the required diagnostic feature values.
2.  Submit the input.
3.  Pass the values through the saved preprocessing and model pipeline.
4.  Receive a **Benign** or **Malignant** prediction.
5.  View the model's prediction probability.

The trained pipeline is stored in:

``` text
breast_cancer_pipeline.pkl
```

------------------------------------------------------------------------

## 📁 Repository Structure

``` text
breast-cancer-streamlit/
│
├── app.py
├── breast_cancer_pipeline.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

### File Description

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `app.py`                            Streamlit application

  `breast_cancer_pipeline.pkl`        Saved StandardScaler + PCA +
                                      Logistic Regression pipeline

  `requirements.txt`                  Python dependencies

  `.gitignore`                        Prevents files such as `.venv` from
                                      being committed

  `README.md`                         Project documentation
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   **Python**
-   **NumPy**
-   **Pandas**
-   **Scikit-learn**
-   **Joblib**
-   **Streamlit**

------------------------------------------------------------------------

## 💻 Run Locally

Clone the repository:

``` bash
git clone https://github.com/Aditya07/breast-cancer-streamlit.git
cd breast-cancer-streamlit
```

Create a virtual environment:

``` bash
python -m venv .venv
```

Activate it on Windows:

``` bash
.venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run the Streamlit application:

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 🔎 Key Learnings

This project helped explore how different stages of a machine learning
workflow affect model performance:

-   Standardization is important when working with models such as
    Logistic Regression and KNN.
-   PCA can reduce dimensionality while retaining most of the
    information in the dataset.
-   KNN performance depends strongly on the choice of `n_neighbors`.
-   `GridSearchCV` provides a systematic approach to hyperparameter
    selection using cross-validation.
-   Multiple evaluation metrics provide a better understanding of
    classification performance than accuracy alone.
-   A complete preprocessing and model pipeline can be saved and reused
    during deployment.
-   Machine learning models can be integrated into an interactive
    application using Streamlit.

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Compare additional classification algorithms.
-   Experiment with different PCA variance thresholds.
-   Perform broader hyperparameter searches.
-   Add model-performance visualizations to the Streamlit app.
-   Improve input validation and user guidance.
-   Add automated tests for the prediction pipeline.
-   Add model versioning and automated retraining.

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is intended **for educational and demonstration purposes
only**.

It is not a medical diagnostic system and should not be used to make
medical decisions or replace evaluation by qualified healthcare
professionals.

------------------------------------------------------------------------

## 👨‍💻 Author

**Advitya07**

Built as part of a machine learning learning journey, covering the
workflow from preprocessing and model evaluation to deployment.

------------------------------------------------------------------------

⭐ If you found this project useful, consider giving the repository a
star!
