"""Preprocessing for the Telco Customer Churn dataset (CM2604, Part 01).

Used by the modelling notebooks so that every model gets exactly the same
cleaning, split and encoding/scaling steps.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# These columns contain "No internet service" / "No phone service".
# That information is already held by InternetService / PhoneService,
# so it is recoded to "No" to remove the redundant category.
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and apply the data-cleaning decisions from the EDA."""
    df = pd.read_csv(path)

    # customerID is a unique identifier with no predictive value
    df = df.drop(columns=["customerID"])

    # TotalCharges is text because 11 rows hold a blank string. All 11 have
    # tenure == 0 (new customers, not billed yet), so 0 is the logical value.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Remove the redundant "No internet service" / "No phone service" categories
    df[INTERNET_ADDONS] = df[INTERNET_ADDONS].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # SeniorCitizen is already 0/1 but is a category, not a quantity
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Target: 1 = churned, 0 = stayed
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def split_data(df: pd.DataFrame):
    """Stratified 80/20 train/test split (keeps the 73.5/26.5 class ratio)."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            stratify=y, random_state=RANDOM_STATE)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical columns.

    Returned unfitted: it must be fitted on the training data only (ideally
    inside a sklearn Pipeline) so that no test-set information leaks in.
    """
    categorical_cols = [c for c in X.columns if c not in NUMERIC_COLS]
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                              sparse_output=False), categorical_cols),
    ])
