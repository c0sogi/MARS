import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import SiameseRoBERTa
from library.trainer import Trainer


def main():
    # ==========================================
    # 1. Setup & Initialization
    # ==========================================
    # Create necessary directories (working, submission)
    Config.create_dirs()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    print("Initializing pipeline...")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Load DataLoaders using the provided library function.
    # We use load_cached_data=True to leverage pre-processed files if available.
    # We do not use debug mode to ensure we train on the full (but small) dataset for best results.
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # ==========================================
    # 3. Model & Trainer Initialization
    # ==========================================
    print("Initializing model and trainer...")
    model = SiameseRoBERTa()
    trainer = Trainer(model, train_loader, val_loader, test_loader)

    # ==========================================
    # 4. Training
    # ==========================================
    print("Starting training...")
    # The fit method handles the training loop, validation per epoch, and early stopping.
    trainer.fit()

    # ==========================================
    # 5. Validation Assessment
    # ==========================================
    print("Performing final validation...")
    # The trainer automatically reloads the best model state after fit() completes.
    val_score = trainer.validate()

    # Print the metric in the strictly required format
    print(f"Final Validation Metric: {val_score}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")
    try:
        # Generate predictions on the validation set
        val_preds = trainer.predict(val_loader)

        # Load validation metadata to get ground truth and features
        val_df = pd.read_csv(Config.VAL_PATH)

        # Ensure alignment between predictions and dataframe
        if len(val_preds) != len(val_df):
            print(
                f"Warning: Validation size mismatch. Preds: {len(val_preds)}, DF: {len(val_df)}"
            )
            min_len = min(len(val_preds), len(val_df))
            val_preds = val_preds[:min_len]
            val_df = val_df.iloc[:min_len]

        val_targets = val_df[Config.TARGET_COLS].values

        # Compute Mean Absolute Error (MAE) per row (sample)
        # This represents the "magnitude of error" for each QA pair
        row_mae = np.mean(np.abs(val_preds - val_targets), axis=1)

        # Extract features for correlation analysis: Text Lengths
        val_df["q_len"] = val_df["question_body"].fillna("").astype(str).apply(len)
        val_df["a_len"] = val_df["answer"].fillna("").astype(str).apply(len)

        # Compute Spearman correlation between Error and Text Lengths
        corr_q, _ = spearmanr(row_mae, val_df["q_len"])
        corr_a, _ = spearmanr(row_mae, val_df["a_len"])

        print(f"Correlation between Error (MAE) and Question Length: {corr_q:.4f}")
        print(f"Correlation between Error (MAE) and Answer Length: {corr_a:.4f}")

    except Exception as e:
        print(f"An error occurred during failure analysis: {e}")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    THRESHOLD = 0.2078132531759014

    if val_score > THRESHOLD:
        print(
            f"\nValidation score {val_score} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        try:
            # Generate predictions on the test set
            test_preds = trainer.predict(test_loader)

            # Retrieve QA IDs for the test set
            # We prefer loading from the cache created by dataset.py to ensure alignment
            test_qa_ids_path = os.path.join(Config.WORKING_DIR, "test_qa_ids.npy")
            if os.path.exists(test_qa_ids_path):
                test_qa_ids = np.load(test_qa_ids_path)
            else:
                # Fallback to reading the CSV directly
                test_df = pd.read_csv(Config.TEST_PATH)
                test_qa_ids = test_df["qa_id"].values

            # Verify dimensions
            if len(test_preds) != len(test_qa_ids):
                raise ValueError(
                    f"Shape mismatch: {len(test_preds)} predictions vs {len(test_qa_ids)} IDs"
                )

            # Construct Submission DataFrame
            submission_df = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
            submission_df.insert(0, "qa_id", test_qa_ids)

            # Save to disk
            submission_df.to_csv(Config.SUBMISSION_SAVE_PATH, index=False)
            print(f"Submission saved successfully to {Config.SUBMISSION_SAVE_PATH}")

        except Exception as e:
            print(f"An error occurred during submission generation: {e}")
    else:
        print(
            f"\nValidation score {val_score} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
