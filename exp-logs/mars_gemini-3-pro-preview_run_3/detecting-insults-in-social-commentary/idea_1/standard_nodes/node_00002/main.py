import pandas as pd
import numpy as np
import os
import sys

# Import classes and functions from the provided library files
from library.config import Config
from library.model import NBLRModel
from library.data_loader import load_datasets
from library.utils import set_seed


def main():
    # Ensure reproducibility
    set_seed(Config.SEED)

    # 1. Load Datasets
    # Using load_cached_data=True to utilize preprocessed parquet files if available
    print("Loading datasets...")
    train_df, val_df, test_df = load_datasets(load_cached_data=True)

    # 2. Initialize and Train Model
    # The NBLRModel combines TF-IDF vectorization (Word & Char) with Naive Bayes weighting and Logistic Regression
    print("Initializing and training model...")
    model = NBLRModel()

    # The fit method trains the model and returns the AUC on the provided validation set
    val_auc = model.fit(train_df, val_df)

    # Print the final validation metric in the required format
    # Full precision is preserved
    print(f"Final Validation Metric: {val_auc}")

    # 3. Failure Analysis
    # We analyze the correlation between prediction error and input text characteristics (length)
    print("\nPerforming Failure Analysis on Validation Set...")

    # Generate probability predictions for the validation set
    # NBLRModel.predict_proba handles the full transformation pipeline
    val_preds = model.predict_proba(val_df)
    val_targets = val_df[Config.LABEL_COL].values

    # Calculate error magnitude (absolute difference between truth and prediction)
    errors = np.abs(val_targets - val_preds)

    # Extract meta-features from validation text
    val_text = val_df[Config.TEXT_COL].astype(str)
    char_lengths = val_text.apply(len)
    word_counts = val_text.apply(lambda x: len(x.split()))

    # Calculate correlations
    corr_char = np.corrcoef(char_lengths, errors)[0, 1]
    corr_word = np.corrcoef(word_counts, errors)[0, 1]

    print("Correlation between Model Error and Input Features:")
    print(f"  Character Length: {corr_char:.6f}")
    print(f"  Word Count:       {corr_word:.6f}")

    # 4. Generate Submission
    print("\nGenerating predictions for Test Set...")
    test_preds = model.predict_proba(test_df)

    # Construct submission DataFrame
    # Load raw test data to preserve original text formatting for submission
    submission_df = pd.read_csv(Config.TEST_DATA_PATH)
    submission_df["Insult"] = test_preds

    # Select and order columns as per sample submission: Insult, Date, Comment
    cols_to_save = ["Insult", Config.DATE_COL, Config.TEXT_COL]
    submission_df = submission_df[cols_to_save]

    # Save to disk
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Process completed successfully.")


if __name__ == "__main__":
    main()
