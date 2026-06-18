# Gender Prediction from Full Name using Machine Learning

## Project Overview

This project predicts gender (Male/Female) from a person's full name 
(First Name + Last Name) using Machine Learning techniques.

The project uses Natural Language Processing (NLP) techniques to extract 
patterns from names and a Machine Learning classification algorithm to 
predict gender.

The model is trained on Indian and foreign name datasets with gender labels.

## Technologies Used

- Python
- Machine Learning
- Natural Language Processing (NLP)
- Scikit-Learn
- Pandas
- NumPy
- Jupyter Notebook


## Machine Learning Approach

The project follows these steps:

1. Data Collection
2. Data Preprocessing
3. Feature Extraction using TF-IDF
4. Model Training using Logistic Regression
5. Gender Prediction
6. Model Evaluation


## Folder Structure
```
Gender-Prediction-Project/
│
├── Data/
│ │
│ ├── data.csv
│ ├── Gender_Data.csv
│ └── test_names.csv
│
├── models/
│ │
│ ├── gender_model.pkl
│ └── vectorizer.pkl
│
├── Gender_Prediction_Final.ipynb
│
├── requirements.txt
│
└── README.md
```



# Dataset

| File Name | Description | Size | Link |
|-----------|-------------|------|------|
| data.csv | Main dataset containing names and gender information with Male label as 0 and Female label as 1 | 3687 KB | [Data Link](Data/data.csv) |
| Gender_Data.csv | Additional dataset containing names and gender labels with Male label as M and Female label as F | 635 KB | [Data Link](Data/Gender_Data.csv) |
| test_names.csv | Testing dataset containing sample names for prediction testing | 143 KB | [Data Link](Data/test_names.csv) |


# Logistic Regression

Logistic Regression is a supervised machine learning classification algorithm
used to predict gender categories (Male/Female) based on a person's full name.

The model learns patterns from numerical features generated using TF-IDF 
Vectorization and performs classification.

## Model Configuration

- Algorithm: Logistic Regression
- C = 3
- Solver = liblinear
- Maximum Iteration = 1000
- Class Weight = Balanced


## Why Logistic Regression?

- Works efficiently for text classification problems.
- Handles binary classification (Male/Female).
- Provides probability scores for prediction confidence.
- Performs well with TF-IDF features.


## Logistic Regression Code Example

```python
from sklearn.linear_model import LogisticRegression


# Create Logistic Regression Model

model = LogisticRegression(
    C=3,
    solver='liblinear',
    max_iter=1000,
    class_weight='balanced'
)


# Train Model

model.fit(
    x_train,
    y_train
)
```
# TF-IDF Vectorization (Feature Extraction)
TF-IDF (Term Frequency-Inverse Document Frequency) is an NLP technique
used to convert names into numerical features.

Machine Learning models cannot directly understand text data, so TF-IDF
converts names into mathematical vectors.

In this project, character-level TF-IDF is used.

Character Level TF-IDF

Example:

Input:

Rahul

Generated Character N-Grams:

ra
ah
hu
ul

rah
ahu
hul

These patterns help the model identify name structures.

N-Gram Range

The project uses:

ngram_range=(2,5)

It extracts:

2 character patterns
3 character patterns
4 character patterns
5 character patterns
TF-IDF Code Example
from sklearn.feature_extraction.text import TfidfVectorizer


# Character TF-IDF Vectorizer

vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2,5)
)


# Convert names into numerical features

X = vectorizer.fit_transform(
    data['Name']
)


