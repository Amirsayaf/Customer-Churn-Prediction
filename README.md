
# Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to leave a telecommunications company.

This project uses customer information and machine learning classification models to identify potential churners and help businesses improve customer retention.

## Dataset

The dataset contains customer information from a telecommunications company.

- Total Records: 7,043
- Input Features: 19
- Target Variable: `Churn`
- Target Classes: `Yes` and `No`

The goal is to predict whether a customer will leave the company.

## Machine Learning Models

The following classification algorithms were implemented and evaluated:

- Logistic Regression
- Random Forest Classifier

Both models were trained using the same dataset and preprocessing pipeline.

## Model Evaluation

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score

Logistic Regression achieved better performance than Random Forest in the current experiment.

The decision threshold was adjusted to 0.20 to prioritize Recall and identify more potential customers who may leave the company.

## Installation & Usage

1. Clone the repository:

```bash
git clone https://github.com/Amirsayaf/Customer-Churn-Prediction.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the training script:

```bash
python src/train.py
```
