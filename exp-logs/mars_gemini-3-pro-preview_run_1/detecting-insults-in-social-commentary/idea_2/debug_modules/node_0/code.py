import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import logging
from sklearn.metrics import roc_auc_score

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import Config
from library.data_loader import load_data
from library.nbsvm_model import run_nbsvm
from library.transformer_model import run_transformer


def setup_environment():
    """
    Sets up the environment for the run:
    - Suppresses warnings
    - Sets seeds
    - Overrides Config for a fast demonstration run
    """
    # Suppress warnings
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Suppress HuggingFace Transformers logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Set seeds
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    # Override Config for Speed/Demo
    print("Overriding Config parameters for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Small subset for speed
    Config.EPOCHS = 1  # Single epoch for demo
    Config.TRAIN_BATCH_SIZE = 8  # Small batch size
    Config.VALID_BATCH_SIZE = 16
    Config.NBSVM_MAX_ITER = 100  # Limit iterations for NBSVM

    # Ensure working directory exists (Config does this, but good to double check)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


def main():
    setup_environment()

    print("\n" + "=" * 40)
    print("1. Data Loading")
    print("=" * 40)

    # Load data using the library function
    # Note: Config.DEBUG is True, so this will load a subset
    train_df, val_df, test_df = load_data(load_cached_data=False)

    # Validation: Check data shapes
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    assert len(train_df) <= Config.DEBUG_SAMPLE_SIZE, "Train data exceeds debug size"
    assert len(val_df) <= Config.DEBUG_SAMPLE_SIZE, "Val data exceeds debug size"
    assert (
        "clean_comment" in train_df.columns
    ), "Preprocessing failed (clean_comment missing)"

    print("\n" + "=" * 40)
    print("2. Running Structural Branch (NBSVM)")
    print("=" * 40)

    # Run NBSVM pipeline
    # We pass load_cached_data=False to force re-computation for the demo
    nbsvm_val_preds, nbsvm_test_preds = run_nbsvm(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validation: Check NBSVM outputs
    assert len(nbsvm_val_preds) == len(val_df), "NBSVM val preds length mismatch"
    assert len(nbsvm_test_preds) == len(test_df), "NBSVM test preds length mismatch"
    assert np.all(
        (nbsvm_val_preds >= 0) & (nbsvm_val_preds <= 1)
    ), "NBSVM preds out of range [0,1]"

    print(
        f"NBSVM Val AUC (Subset): {roc_auc_score(val_df['Insult'], nbsvm_val_preds):.4f}"
    )

    print("\n" + "=" * 40)
    print("3. Running Semantic Branch (Transformer)")
    print("=" * 40)

    # Run Transformer pipeline
    # This handles tokenization, dataset creation, and training loop
    transformer_val_preds, transformer_test_preds = run_transformer(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validation: Check Transformer outputs
    assert len(transformer_val_preds) == len(
        val_df
    ), "Transformer val preds length mismatch"
    assert len(transformer_test_preds) == len(
        test_df
    ), "Transformer test preds length mismatch"
    # Note: Transformer output is sigmoid, so strictly [0, 1]
    assert np.all(
        (transformer_val_preds >= 0) & (transformer_val_preds <= 1)
    ), "Transformer preds out of range"

    print(
        f"Transformer Val AUC (Subset): {roc_auc_score(val_df['Insult'], transformer_val_preds):.4f}"
    )

    print("\n" + "=" * 40)
    print("4. Ensembling and Evaluation")
    print("=" * 40)

    # Weighted Average Ensemble
    w_trans = Config.TRANSFORMER_WEIGHT
    w_nbsvm = Config.NBSVM_WEIGHT

    print(f"Ensemble Weights -> Transformer: {w_trans}, NBSVM: {w_nbsvm}")

    ensemble_val_preds = (w_trans * transformer_val_preds) + (w_nbsvm * nbsvm_val_preds)
    ensemble_test_preds = (w_trans * transformer_test_preds) + (
        w_nbsvm * nbsvm_test_preds
    )

    # Calculate Final Validation AUC
    final_auc = roc_auc_score(val_df["Insult"], ensemble_val_preds)
    print(f"Final Ensemble Validation AUC: {final_auc:.4f}")

    # Basic sanity check that ensemble improves or stays reasonable
    # (On a tiny debug subset, improvement isn't guaranteed, but code execution is verified)
    assert 0 <= final_auc <= 1, "AUC calculation error"

    print("\n" + "=" * 40)
    print("5. Generating Submission")
    print("=" * 40)

    # Load sample submission to get correct format/IDs if necessary
    # The provided sample_submission_null.csv has columns: Insult, Date, Comment
    # However, usually Kaggle-style submissions need an ID.
    # Based on the task description: "Your predictions should be a number in the range [0,1]. See 'sample_submission_null.csv' for the correct format."
    # The sample submission provided in the prompt description has columns: Insult, Date, Comment.
    # It seems we need to fill the 'Insult' column with probabilities.

    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Since we ran on a subset (DEBUG mode), we can't generate a full valid submission
    # for the actual leaderboard if we only have 200 predictions.
    # However, for this code demonstration, we will demonstrate how to map the predictions back.

    # IMPORTANT: In a real run (Config.DEBUG = False), len(test_df) == len(sample_sub).
    # Here, we will create a placeholder submission dataframe matching the test_df size
    # to demonstrate the logic, then save it.

    submission_df = test_df.copy()
    submission_df["Insult"] = ensemble_test_preds

    # Select columns as per sample submission
    # If sample submission has specific columns, we should match them.
    # The prompt says sample_submission_null.csv has columns: Insult, Date, Comment.
    cols_to_keep = ["Insult", "Date", "Comment"]
    submission_df = submission_df[cols_to_keep]

    # Save submission
    save_path = Config.SUBMISSION_PATH
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved to: {save_path}")
    print("First 5 rows of submission:")
    print(submission_df.head())

    # Verify file existence
    assert os.path.exists(save_path), "Submission file was not created"

    print("\nExecution completed successfully.")


if __name__ == "__main__":
    main()
