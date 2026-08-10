import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib



# Load the original dataset
df = pd.read_csv("datasets/training.csv")


# Display the dataset
print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)


print("\nDataset Columns:")
print(df.columns.tolist())

#MISSING VALUES
print("\nMissing Values:")
print(df.isnull().sum())

#FILL CATEGORICAL MISSING VALUES
categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed"
]

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

#FILL NUMERICAL MISSING VALUES
numerical_columns = [
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())

# CHECK MISSING VALUES AGAIN
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

#Remove Unnecessary Column
df = df.drop("Loan_ID", axis=1)

print("\nColumns after removing Loan_ID:")
print(df.columns.tolist())

#CREATE FEATURES (X)

X = df.drop("Loan_Status", axis=1)

print("\nFeatures (X):")
print(X.head())

#CREATE TARGET (y)
y = df["Loan_Status"]

print("\nTarget (y):")
print(y.head())

#CHECK X AND y SHAPE
print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

#ENCODE TARGET VARIABLE
y = y.map({
    "N": 0,
    "Y": 1
})

print("\nEncoded Target Values:")
print(y.head())

print("\nTarget Value Counts:")
print(y.value_counts())

#Identify categorical columns
categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

print("\nCategorical Columns:")
print(categorical_columns)

#Identify numerical columns

numerical_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

print("\nNumerical Columns:")
print(numerical_columns)

#CREATE PREPROCESSOR

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

print("\nPreprocessor created successfully!")

# Split the Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData Split Successfully!")

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

# PREPROCESS TRAINING DATA
X_train_processed = preprocessor.fit_transform(X_train)

print("\nTraining data preprocessing completed!")
print("X_train_processed shape:", X_train_processed.shape)

# PREPROCESS TESTING DATA
X_test_processed = preprocessor.transform(X_test)

print("\nTesting data preprocessing completed!")
print("X_test_processed shape:", X_test_processed.shape)

#TRAIN XGBOOST MODEL
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    eval_metric="logloss"
)


# Train the model
xgb_model.fit(
    X_train_processed,
    y_train
)

print("\nXGBoost model trained successfully!")

# XGBOOST PREDICTIONS

y_pred = xgb_model.predict(X_test_processed)

print("\nPredictions made successfully!")

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

print("\nFirst 10 Predicted Values:")
print(y_pred[:10])

# XGBOOST MODEL EVALUATION
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Calculate metrics
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# Display results
print("\nXGBOOST MODEL RESULTS")
print("=====================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# SAVE MODEL AND PREPROCESSOR
joblib.dump(
    xgb_model,
    "model/xgboost_model.pkl"
)

print("\nXGBoost model saved successfully!")

joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)

print("Preprocessor saved successfully!")
