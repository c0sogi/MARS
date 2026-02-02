import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import torch

# Import from provided library files
from library.config import (
    seed_everything,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_FEATS_CACHE,
    VAL_FEATS_CACHE,
    TEST_FEATS_CACHE,
    SUBMISSION_PATH,
    WORKING_DIR,
)
from library.preprocessor import preprocess_metadata
from library.dataset import get_dataloaders
from library.classifier import MalignancyClassifier
from library.utils import save_submission


def run_failure_analysis(df_val, y_val, val_probs):
    """
    Performs failure analysis by correlating prediction errors with input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate absolute error
    errors = np.abs(y_val - val_probs)

    # Create a temporary dataframe for analysis
    analysis_df = df_val.copy()
    analysis_df["error"] = errors

    # 1. Correlation with Age (Numerical)
    # Fill NaNs with mean for correlation calculation
    if "age_approx" in analysis_df.columns:
        mean_age = analysis_df["age_approx"].mean()
        analysis_df["age_approx_filled"] = analysis_df["age_approx"].fillna(mean_age)
        corr_age = analysis_df["age_approx_filled"].corr(analysis_df["error"])
        print(f"Correlation (Error vs Age): {corr_age}")

    # 2. Correlation with Sex (Categorical -> Binary)
    if "sex" in analysis_df.columns:
        # Map to numeric: male=0, female=1, others=NaN
        analysis_df["sex_mapped"] = analysis_df["sex"].map({"male": 0, "female": 1})
        # Compute correlation on non-null values
        if analysis_df["sex_mapped"].notna().any():
            corr_sex = analysis_df["sex_mapped"].corr(analysis_df["error"])
            print(f"Correlation (Error vs Sex): {corr_sex}")

    # 3. Correlation with Anatomy (Categorical -> Label Encoded)
    if "anatom_site_general_challenge" in analysis_df.columns:
        # Handle NaNs as a category
        analysis_df["anatom_filled"] = analysis_df[
            "anatom_site_general_challenge"
        ].fillna("missing")
        le = LabelEncoder()
        analysis_df["anatom_enc"] = le.fit_transform(
            analysis_df["anatom_filled"].astype(str)
        )
        corr_anatom = analysis_df["anatom_enc"].corr(analysis_df["error"])
        print(f"Correlation (Error vs Anatom Site): {corr_anatom}")

    # 4. Correlation with Target Class
    # Does the model error more on malignant (1) or benign (0)?
    corr_target = pd.Series(y_val).corr(pd.Series(errors))
    print(f"Correlation (Error vs Target Class): {corr_target}")


def main():
    # 1. Setup
    seed_everything()
    print("Starting pipeline...")

    # 2. Preprocess Metadata
    # This generates/loads tabular features for train, val, and test
    print("Preprocessing metadata...")
    X_tab_train, X_tab_val, X_tab_test = preprocess_metadata(
        TRAIN_CSV, VAL_CSV, TEST_CSV, cache_dir=WORKING_DIR, load_cached_data=True
    )

    # Load raw dataframes (needed for file paths in dataloaders)
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # 3. Prepare DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        df_train, df_val, df_test, X_tab_train, X_tab_val, X_tab_test
    )

    # 4. Model Training
    # Get tabular dimension from the processed data
    tabular_dim = X_tab_train.shape[1]

    print("Initializing MalignancyClassifier (End-to-End Fine-Tuning)...")
    clf = MalignancyClassifier(tabular_dim)

    # Train directly with dataloaders
    clf.fit(train_loader, val_loader)

    # 5. Validation & Metrics
    print("Performing Validation...")
    # We need to extract targets from val_loader for metric calculation
    # (predict_proba now takes a loader)
    val_probs = clf.predict_proba(val_loader)

    # Re-extract y_val from the dataframe or loader to ensure alignment
    y_val = df_val["target"].values

    final_auc = roc_auc_score(y_val, val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    run_failure_analysis(df_val, y_val, val_probs)

    # 8. Submission
    print("Generating Submission...")
    test_probs = clf.predict_proba(test_loader)

    # Ensure we use the image names from the test dataframe
    test_image_names = df_test["image_name"].values
    save_submission(test_image_names, test_probs, SUBMISSION_PATH)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
