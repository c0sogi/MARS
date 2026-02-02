import os
import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import log_loss

# Import library modules
from library.configuration import Config
from library.utilities import seed_everything, compute_log_loss
from library.transformer_expert import train_transformer_expert
from library.linear_expert import train_linear_expert
from library.meta_learner import train_predict_xgboost

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)

    # 2. Configure for Fast Baseline
    # Reduce epochs to ensure completion within time limits
    Config.EPOCHS = 2
    # Adjust batch size for A100 (40GB).
    # Default was 4, increasing to 6 for speed, adjusting accum steps to maintain effective batch size ~18
    Config.TRAIN_BATCH_SIZE = 6
    Config.GRADIENT_ACCUMULATION_STEPS = 3

    # 3. Prepare Combined Validation and Test Set
    # The provided library functions are designed to predict on a single file defined in Config.TEST_DATA_PATH.
    # To get predictions for both the hold-out validation set (for metrics) and the test set (for submission)
    # without retraining, we combine them into a single temporary file.

    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Concatenate ID and Text columns (Target 'author' is not needed for inference)
    combined_df = pd.concat(
        [val_df[["id", "text"]], test_df[["id", "text"]]], axis=0
    ).reset_index(drop=True)

    combined_path = os.path.join(Config.WORKING_DIR, "combined_val_test.csv")
    combined_df.to_csv(combined_path, index=False)

    # Override Config to point to the combined file
    Config.TEST_DATA_PATH = combined_path

    print(
        f"Combined dataset created at {combined_path} with {len(combined_df)} samples."
    )
    print("Starting Expert Training...")

    # 4. Train Experts
    # We pass load_cached_data=False to force the models to predict on our new combined dataset

    # Transformer Expert (DeBERTa-v3-Large)
    # Returns: OOF predictions (Train), Combined Predictions (Val + Test)
    trans_oof, trans_combined = train_transformer_expert(load_cached_data=False)

    # Linear Expert (TF-IDF + Logistic Regression)
    lin_oof, lin_combined = train_linear_expert(load_cached_data=False)

    # 5. Meta-Learner (XGBoost)
    # Trains on OOF predictions, predicts on Combined set
    # We disable automatic submission saving here to handle it conditionally later
    print("Training Meta-Learner...")
    submission_combined = train_predict_xgboost(
        trans_oof, trans_combined, lin_oof, lin_combined, save_submission=False
    )

    # 6. Split Predictions
    n_val = len(val_df)

    # The first n_val rows correspond to the validation set
    val_preds_df = submission_combined.iloc[:n_val].copy()
    # The remaining rows correspond to the test set
    test_preds_df = submission_combined.iloc[n_val:].copy()

    # 7. Validation Assessment
    # Load true labels
    y_true = val_df["author"].map(Config.LABEL2ID).values
    y_pred = val_preds_df[["EAP", "HPL", "MWS"]].values

    # Compute Metric
    val_loss = compute_log_loss(y_true, y_pred)
    print(f"Final Validation Metric: {val_loss}")

    # 8. Failure Analysis
    print("\nFailure Analysis:")

    # Calculate per-sample Log Loss (Cross Entropy)
    # Clip probabilities to avoid log(0)
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

    # Create one-hot encoding of true labels
    y_true_onehot = np.zeros_like(y_pred)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1

    # Compute negative log likelihood per sample
    sample_losses = -np.sum(y_true_onehot * np.log(y_pred_clipped), axis=1)

    # Extract meta-features for correlation analysis
    texts = val_df["text"].fillna("").astype(str)
    char_lens = texts.apply(len)
    word_counts = texts.apply(lambda x: len(x.split()))

    analysis_df = pd.DataFrame(
        {"loss": sample_losses, "char_len": char_lens, "word_count": word_counts}
    )

    # Compute correlation
    corr = analysis_df.corr()["loss"].drop("loss")
    print("Correlation between Error (LogLoss) and Features:")
    print(corr)

    # 9. Submission Generation
    THRESHOLD = 0.25336663725445785

    if val_loss < THRESHOLD:
        print(
            f"\nValidation score {val_loss} meets threshold {THRESHOLD}. Generating submission..."
        )

        # Ensure the submission directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Save the test portion of the predictions
        test_preds_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation score {val_loss} does not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
