import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = os.path.join(
    BASE_DIR,
    "training.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "xgboost_model.pkl"
)

PREPROCESSOR_FILE = os.path.join(
    BASE_DIR,
    "preprocessor.pkl"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOAN APPROVAL PREDICTION - MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(
    f"Dataset loaded successfully: {df.shape}"
)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# REMOVE LOAN ID
# ============================================================

if "Loan_ID" in df.columns:

    df = df.drop(
        "Loan_ID",
        axis=1
    )

    print("\nLoan_ID column removed.")


# ============================================================
# TARGET COLUMN
# ============================================================

target_column = "Loan_Status"

if target_column not in df.columns:

    raise ValueError(
        f"'{target_column}' column was not found "
        "in the dataset."
    )


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    target_column,
    axis=1
)

y = df[target_column].copy()


# ============================================================
# CLEAN TARGET
# ============================================================

print("\nCleaning target column...")

y = (
    y.astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# CONVERT Y/N TO 1/0
# ============================================================

y = y.map({
    "Y": 1,
    "N": 0
})


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

if y.isna().any():

    print(
        "\nUnexpected Loan_Status values found:"
    )

    print(
        df.loc[
            y.isna(),
            target_column
        ].unique()
    )

    raise ValueError(
        "Loan_Status contains values other than Y/N."
    )


# ============================================================
# CONVERT TARGET TO INTEGER
# ============================================================

y = y.astype(int)


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numeric_columns)


# ============================================================
# NUMERICAL PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# ============================================================
# PREPROCESS DATA
# ============================================================

print("\nPreprocessing data...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost model...")


model = XGBClassifier(

    n_estimators=200,

    max_depth=4,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train_processed,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_processed
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)

print(
    f"MODEL ACCURACY: {accuracy * 100:.2f}%"
)

print("=" * 60)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving XGBoost model...")

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

print("Saving preprocessor...")

joblib.dump(
    preprocessor,
    PREPROCESSOR_FILE
)


# ============================================================
# SUCCESS
# ============================================================

print("\n" + "=" * 60)

print("MODEL FILES CREATED SUCCESSFULLY!")

print("=" * 60)

print(
    f"\nXGBoost model:"
    f"\n{MODEL_FILE}"
)

print(
    f"\nPreprocessor:"
    f"\n{PREPROCESSOR_FILE}"
)

print("\nFiles are saved directly inside:")

print(BASE_DIR)

print("\nYou can now run:")

print(
    "\nstreamlit run app.py"
)

print()