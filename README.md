# Gender Prediction from Full Name using Machine Learning

## Project Overview

This project predicts gender (Male/Female) from a person's full name
using Machine Learning.

Technologies: - Logistic Regression - TF-IDF Character N-Gram

## Project Structure
```
Gender-Prediction-Project/

│
├── Data/
│ │
│ ├── data.csv
│ └── Gender_Data.csv
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

## Dataset

| File Name | Description | Size | Link |
|-----------|-------------|------|------|
| data.csv | Main dataset containing names and gender information with Male label as 0 and Female as 1 | 3687 KB | [Data Link](Data/data.csv) |
| Gender_Data.csv | Additional gender dataset containing names and labels with Male label as M and Female as F | 635 KB | [Data Link](Data/Gender_Data.csv) |
| test_names.csv | Testing dataset containing sample names for model prediction testing | 143 KB | [Data Link](Data/test_names.csv) |

## Model

Logistic Regression with TF-IDF Character N-Gram features.

## Prediction Example

{ 'input': 'Ansh Tyagi', 'predicted_gender': 'Male', 'confidence':
'high', 'probability': 0.94, 'all_probabilities': { 'Female': 0.06,
'Male': 0.94 } }

## Sample Predictions

Bhavya Gupta: - Predicted Gender: Male - Confidence: Low - Probability:
0.52

Ansh Tyagi: - Predicted Gender: Male - Confidence: High - Probability:
0.94

Kumar Ambuj: - Predicted Gender: Male - Confidence: High - Probability:
0.97

## Evaluation

Tested on 10,000 names.

Total Gender Prediction Accuracy:

89.47%

## Confidence Levels

\< 0.55 : Low\
0.55 - 0.70 : Medium\
\> 0.70 : High

## Installation

pip install -r requirements.txt

Run:

jupyter notebook

Open:

Gender_Prediction_Final.ipynb

## Future Improvements

-   Deep Learning based prediction
-   Streamlit Web Application
-   REST API Deployment
-   Multilingual name support
