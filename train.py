
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)


# Load dataset

data = pd.read_csv(
    'WA_Fn-UseC_-Telco-Customer-Churn.csv'
)


# Separate features and target

X = data.drop(['Churn', 'customerID'], axis=1)
y = data['Churn']


# Convert TotalCharges to numeric

X['TotalCharges'] = pd.to_numeric(
    X['TotalCharges'],
    errors='coerce'
)


# Fill missing values

X['TotalCharges'] = X['TotalCharges'].fillna(0)


print('X shape:', X.shape)
print('y shape:', y.shape)


# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# Identify numerical and categorical columns

numeric_features = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

categorical_features = X.select_dtypes(
    include=['object']
).columns.tolist()

print('Numeric features:', numeric_features)
print('Categorical features:', categorical_features)


# Create preprocessing transformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


# Create machine learning pipeline

model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ]
)


# Train the model
rf_model= Pipeline(steps=[('preprocessor', preprocessor),('classifier',RandomForestClassifier(
n_estimators=100,
random_state=42))])

rf_model.fit(X_train, y_train)

print('Random Forest training completed!')

model.fit(X_train, y_train)

print('Model training completed!')

joblib.dump(
    {
        'model':model,
        'threshold':0.20
    },
    'churn_model.joblib'
)

print('Model saved successfully!')

loaded_artifact=joblib.load('churn_model.joblib')

loaded_model = loaded_artifact['model']
loaded_threshold = loaded_artifact['threshold']

print('Model loaded successfully!')
print('Loaded threshold:', loaded_threshold)

loaded_probabilities = loaded_model.predict_proba(X_test)

loaded_predictions = (
    loaded_probabilities[:, 1] >= loaded_threshold
).astype(int)

print('First 10 loaded predictions:')
print(loaded_predictions[:10])

# Make predictions
y_pred = model.predict(X_test)

rf_pred = rf_model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='Yes')

rf_accuracy = accuracy_score(y_test, rf_pred)

rf_f1 = f1_score(
    y_test,
    rf_pred,
    pos_label='Yes'
)
print('Random Forest Accuracy:', rf_accuracy)
print('Random Forest F1:', rf_f1)

comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [accuracy, rf_accuracy],
    'F1 Score': [f1, rf_f1]
})

print('\nModel Comparison:')
print(comparison)

print('Accuracy:', accuracy)
print('F1 Score:', f1)

# Confiusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['No', 'Yes'])

print(('\nConfusion Matrix:'))
print(cm)

# Classification Report
report = classification_report(
    y_test,
    y_pred,
    labels=['No', 'Yes']
)

print('\nClassification Report:')
print(report)

y_pred = model.predict(X_test)



# Get prediction probabilities
y_prob = model.predict_proba(X_test)

print('\nFirst 5 prediction probabilities:')
print(y_prob[:5])

threshold = 0.30

y_pred_new = (y_prob[:, 1] >= threshold).astype(int)

print('\nPredictions with new threshold:')
print(y_pred_new[:10])

# Change Decision Threshold
threshold = 0.30

y_pred_new = (y_prob[:, 1] >= threshold).astype(int)

# Convert predictions to original labels
y_pred_new = pd.Series(y_pred_new).map({
    0: 'No',
    1: 'Yes'
})

# Evaluate model with new threshold
cm_new = confusion_matrix(
    y_test,
    y_pred_new,
    labels=['No', 'Yes']
)

print('\nConfusion Matrix with new threshold:')
print(cm_new)

# Classification Report with new threshold
report_new = classification_report(
    y_test,
    y_pred_new,
    labels=['No', 'Yes']
)

print('\nClassification Report with new threshold:')
print(report_new)

thresholds = [0.20, 0.30, 0.40]

results = []

for threshold in thresholds:

    y_pred_temp = (y_prob[:, 1] >= threshold).astype(int)

    y_pred_temp = pd.Series(y_pred_temp).map({
        0: 'No',
        1: 'Yes'
    })


    precision = precision_score(
    y_test,
    y_pred_temp,
    pos_label='Yes'
)    

    recall = recall_score(y_test ,y_pred_temp ,pos_label='Yes' )

    f1= f1_score(y_test, y_pred_temp, pos_label='Yes')

    results.append({'threshold': threshold,
    'precision': precision,
    'recall': recall,
    'f1': f1
})

    print(
    f'Threshold: {threshold} | '
    f'Precision: {precision:.3f} | '
    f'Recall: {recall:.3f} | '
    f'F1: {f1:.3f}'
)

    results_df = pd.DataFrame(results)

print('\nFinal Threshold Comparison:')
print(results_df)





