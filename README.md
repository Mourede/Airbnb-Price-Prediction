# Airbnb Superhost Prediction

A machine learning project that predicts whether a New York City Airbnb host is a **Superhost** from listing, pricing, availability and review features.

## Overview

- **Goal:** Predict Superhost status, so hosts can see which factors (reviews, response rate, availability, pricing) are linked to earning it.
- **Data:** `airbnbData_train.csv`, 28,022 NYC listings with 50 preprocessed columns (Inside Airbnb data).
- **Models compared:** K-Nearest Neighbors, Decision Tree and Logistic Regression.
- **Best model:** Logistic Regression (`C=100`, L2 penalty), saved as `best_logistic_regression_model.pkl`.

## Dataset

| Feature group | Example columns |
|---------------|-----------------|
| Host | `host_response_rate`, `host_listings_count`, `n_host_verifications` |
| Property | `accommodates`, `bedrooms`, `bathrooms`, `beds`, `room_type_*` (one-hot) |
| Location | `neighbourhood_group_cleansed_*` (one-hot: Bronx, Brooklyn, Manhattan, Queens, Staten Island) |
| Availability | `availability_30/60/90/365`, `minimum_nights`, `maximum_nights`, `instant_bookable` |
| Reviews | `number_of_reviews`, `review_scores_rating`, `review_scores_location`, `reviews_per_month` |
| Pricing | `price` |
| **Target** | `host_is_superhost` (True/False) |

## Methodology

1. **Preprocessing:** handled missing values with imputation and one-hot encoded borough and room type.
2. **Feature selection:** set `host_is_superhost` as the label and used the other 49 columns as features.
3. **Modeling:** trained KNN, Decision Tree and Logistic Regression classifiers on a train/test split.
4. **Tuning:** optimized hyperparameters with `GridSearchCV` and cross-validation.
5. **Evaluation:** compared models with ROC-AUC and precision-recall analysis, and saved the best one.

## Repository Structure

```
airbnb-superhost-prediction/
├── airbnbData_train.csv                # Preprocessed NYC listings (28,022 rows)
├── best_logistic_regression_model.pkl  # Tuned Logistic Regression model
└── README.md
```

## How to Run

```bash
git clone https://github.com/Mourede/airbnb-superhost-prediction.git
cd airbnb-superhost-prediction
pip install pandas scikit-learn
```

```python
import pickle
import pandas as pd

df = pd.read_csv("airbnbData_train.csv")
with open("best_logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

X = df.drop(columns=["host_is_superhost"])
y = df["host_is_superhost"]
print(model.predict(X.head()))
```

## Tools

Python, pandas, NumPy, scikit-learn, Jupyter Notebook

## Author

**Mourede Otchere** | [LinkedIn](https://www.linkedin.com/in/mouredeo) | [fanel.otchere@gmail.com](mailto:fanel.otchere@gmail.com)
