# Gender Prediction from Full Name using Machine Learning

## Project Overview

This project predicts the gender (Male/Female) of a person based on their full name (First Name + Last Name) using Machine Learning techniques.

The model is trained on a combination of Indian and foreign name datasets collected from Kaggle. Logistic Regression is used as the classification algorithm, while TF-IDF Character N-Grams are used for feature extraction.

The system learns character patterns from names and predicts gender along with confidence scores.

---

## Project Structure

```text
Gender-Prediction-Project/
│
├── Data/
│   ├── data.csv
│   └── Gender_Data.csv
│
├── Gender_Prediction_Final.ipynb
│
├── test.py
|
├── requirements.txt
│
├── .gitignore
|
├── README.md
│
└── models/
    ├── gender_model.pkl
    └── vectorizer.pkl
```

---

## Dataset Information

### Dataset 1

**File Name:** `data.csv`

* Contains names and corresponding gender labels.
* Includes a mixture of Indian and international names.
* Used for training and evaluation.

### Dataset 2

**File Name:** `Gender_Data.csv`

* Additional dataset containing gender-labeled names.
* Combined with Dataset 1 to improve diversity and model performance.

### Data Sources

* Kaggle Gender Prediction Datasets
* Publicly available name datasets

---

## Technologies Used

### Logistic Regression

* Supervised Machine Learning algorithm.
* Used for binary classification (Male/Female).
* Handles class imbalance using:

```python
LogisticRegression(
        C=10,
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear'
    )
)
```

### TF-IDF Vectorization

* Converts names into numerical features.
* Uses character-level analysis.
* Captures meaningful patterns in names.

```python
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3,5),
    sublinear_tf=True,
)
```

---

## Data Preprocessing

The following preprocessing steps are applied:

* Merge multiple datasets.
* Remove duplicates.
* Remove missing values.
* Convert text to lowercase.
* Handle special characters.
* Normalize name formats.
* Filter abnormal text lengths.

---

## Feature Engineering

TF-IDF Character N-Grams are used to transform names into numerical vectors.

Benefits:

* Captures prefixes and suffixes.
* Learns common gender-specific name patterns.
* Improves classification accuracy.

---

## Model Training

### Train-Test Split

```text
Training Data : 80%
Testing Data  : 20%
```

### Model

```python
LogisticRegression(
        C=10,
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear'
    )
```

---

## Prediction Confidence Levels

| Confidence Score | Level             |
| ---------------- | ----------------- |
| < 0.55           | Low Confidence    |
| 0.55 - 0.70      | Medium Confidence |
| > 0.70           | High Confidence   |

---

## Model Performance / Evaluation

The model was evaluated using Accuracy Score and Classification Report to measure its performance on the test dataset.

### Accuracy Score

```text
Accuracy: 0.8843035437733443
(~88.43%)
```

### Classification Report

```text
              precision    recall  f1-score   support

      Female       0.91      0.89      0.90     19207
        Male       0.85      0.88      0.86     13724

    accuracy                           0.88     32931
   macro avg       0.88      0.88      0.88     32931
weighted avg       0.89      0.88      0.88     32931
```

### Performance Summary

The results indicate that the Logistic Regression model combined with TF-IDF Character N-Gram features performs effectively for gender prediction from full names, achieving approximately 89% accuracy on the test dataset.

---

## Installation

### Clone Repository

```bash
git clone <repository-link>
cd Gender-Prediction-Project
```

### Create Virtual Environment

```bash
python -m venv venv_predict
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---


## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/Gender_Prediction.ipynb
```

Run all cells to:

* Load data
* Train model
* Evaluate accuracy
* Predict gender from names

---

## Example Prediction

### Input

```python
predict_name_gender("Ansh Tyagi")
```

### Output

```python
{
    'input': 'Ansh Tyagi',
    'predicted_gender': 'M',
    'confidence': 'high',
    'probability': 0.95,
    'all_probabilities': {
        'F': 0.05,
        'M': 0.95
    }
}
```

### Output Description

- **input** → Name provided by the user.
- **predicted_gender** → Predicted gender (M/F).
- **confidence** → Confidence level (Low, Medium, High).
- **probability** → Probability of the predicted gender.
- **all_probabilities** → Probability scores for all gender classes.

---

## Future Improvements

* Deep Learning based prediction.
* Streamlit Web Application.
* REST API Deployment.
* Multilingual Name Support.
* Real-Time Prediction Dashboard.

---
