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

| File Name | Description | Size |
|-----------|-------------|------|
| data.csv | Main dataset containing names and gender information with Male label as 0 and Female label as 1 | 3687 KB |
| Gender_Data.csv | Additional dataset containing names and gender labels with Male label as M and Female label as F | 635 KB | 
| test_names.csv | Testing dataset containing sample names for prediction testing | 143 KB |


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

TF-IDF (Term Frequency-Inverse Document Frequency) is a Natural Language Processing (NLP) technique used to convert text into numerical features. Because Machine Learning models cannot directly understand raw text, TF-IDF converts names into mathematical vectors.

This project utilizes **character-level TF-IDF** to capture the underlying structure and patterns of names.

### Character-Level TF-IDF Example
When processing the input name **"Rahul"**, the vectorizer breaks it down into smaller character sequences (N-grams):

*   **2-grams:** `ra`, `ah`, `hu`, `ul`
*   **3-grams:** `rah`, `ahu`, `hul`

These structural patterns help the machine learning model identify and learn unique name characteristics.

### N-Gram Range Configuration
The project is configured with `ngram_range=(2,5)`. This setting forces the vectorizer to simultaneously extract:
*   2-character patterns
*   3-character patterns
*   4-character patterns
*   5-character patterns

### Code Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the character-level TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 5)
)

# Convert names into numerical features
X = vectorizer.fit_transform(data['Name'])
```
## Training Process

The dataset is split into a standard train-test ratio to ensure robust model evaluation:
*   **80% Training Data:** Used to train the machine learning model.
*   **20% Testing Data:** Kept hidden during training to evaluate final performance.

---

## Prediction Example

### Sample Input
```text
Ansh Tyagi
```

### Sample Output API Response
```json
{
 "input": "Ansh Tyagi",
 "predicted_gender": "Male",
 "confidence": "High",
 "probability": 0.94,
 "all_probabilities": {
      "Female": 0.06,
      "Male": 0.94
 }
}
```

### Sample Predictions Summary

| Name | Predicted Gender | Confidence | Probability |
| :--- | :--- | :--- | :--- |
| Bhavya Gupta | Male | Low | 0.52 |
| Ansh Tyagi | Male | High | 0.94 |
| Kumar Ambuj | Male | High | 0.97 |

---
### Script Execution Output

Testing the model and find prediction accuracy of the test_names dataset

```text
        Name Actual Gender Predicted Gender
0         John          Male             Male
1         Mary        Female           Female
2      Michael          Male             Male
3     Jennifer        Female           Female
4        David          Male             Male
...        ...           ...              ...
9995     Salih          Male             Male
9996    Tovino          Male             Male
9997    Thomas          Male             Male
9998    Ansaad          Male             Male
9999      Adil          Male             Male

[10000 rows x 3 columns]

Total Gender Prediction Accuracy: 89.47 %
```
## Evaluation

The model was comprehensively tested on a dataset of **10,000 names**.

### Performance Metrics
*   **Accuracy Score:** 89.47%
*   **Precision:** Evaluated per class
*   **Recall:** Evaluated per class
*   **F1-Score:** Evaluated per class
*   **Classification Report:** Generated via Scikit-Learn

### Confidence Levels
The system maps prediction probabilities to three distinct confidence categories:
*   **Low:** Probability `< 0.55`
*   **Medium:** Probability `0.55 - 0.70`
*   **High:** Probability `> 0.70`

---

## Saved Model Files

After successful training, the production-ready assets are saved in the `models/` directory:

```text
models/
├── gender_model.pkl    # Trained Logistic Regression classifier
└── vectorizer.pkl      # Trained character-level TF-IDF Vectorizer
```

---

## Installation & Usage

### 1. Install Dependencies
Install all required libraries using the package manager:
```bash
pip install -r requirements.txt
```

### 2. Run the Project
Launch the Jupyter Notebook environment to interact with the project:
```bash
jupyter notebook
```
Open and execute the cells in: **`Gender_Prediction_Final.ipynb`**

---

## Future Improvements
*   [ ] Implement **Deep Learning** (LSTM/Transformers) based gender prediction.
*   [ ] Build an interactive **Streamlit Web Application** interface.
*   [ ] Deploy the backend as a lightweight **REST API**.
*   [ ] Expand support for **multilingual names**.
*   [ ] Enhance model robustness by improving **dataset diversity**.

---
## License
This project is created strictly for educational purposes.


