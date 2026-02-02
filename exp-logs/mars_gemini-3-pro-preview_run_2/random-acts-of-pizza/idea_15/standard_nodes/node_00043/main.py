import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.trainer import Trainer
from library.data_loader import DataLoader
from library.text_encoder import TextEncoder
from library.tabular_processor import TabularProcessor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Train and Generate Initial Submission
    # The Trainer handles the heavy lifting: CV, model saving, and test inference.
    print("Initializing Trainer...")
    trainer = Trainer()
    trainer.train_and_submit(load_cached_data=True)

    # 3. Validation (Reconstructing OOF for df_val)
    print("\n--- Starting Validation on Hold-out Set ---")

    # Load data to reconstruct splits
    df_train, df_val, df_test = DataLoader.load_data(load_cached_data=True)

    # Reconstruct the 'full' dataset used by Trainer
    # Trainer does: df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    y_full = df_full["requester_received_pizza"].values

    # Identify indices corresponding to the original validation set
    # Since df_val was appended, its indices start after df_train
    val_start_idx = len(df_train)
    val_indices_in_full = np.arange(val_start_idx, len(df_full))

    # Prepare features for split reconstruction (need correct shapes)
    text_encoder = TextEncoder()
    tabular_processor = TabularProcessor()

    # Load embeddings (Trainer already cached these)
    X_text_train = text_encoder.encode(
        df_train, Config.TRAIN_EMBEDDINGS_PATH, load_cached_data=True
    )
    X_text_val = text_encoder.encode(
        df_val, Config.VAL_EMBEDDINGS_PATH, load_cached_data=True
    )
    X_text_full = np.vstack([X_text_train, X_text_val])

    # Load tabular (fast processing)
    X_tab_train = tabular_processor.process(df_train)
    X_tab_val = tabular_processor.process(df_val)
    X_tab_full = np.vstack([X_tab_train, X_tab_val])

    # Replicate StratifiedKFold
    skf = StratifiedKFold(
        n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED
    )

    # Array to store OOF predictions for the validation subset
    # We map back to the original df_val index (0 to len(df_val)-1)
    val_oof_preds = np.zeros(len(df_val))
    val_covered_mask = np.zeros(len(df_val), dtype=bool)

    models_dir = os.path.join(Config.WORKING_DIR, "models")

    print("Reconstructing OOF predictions for validation set...")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_text_full, y_full)):
        # Find which samples in this fold's validation set belong to our hold-out df_val
        # Intersection of test_idx (fold validation) and val_indices_in_full (hold-out set)
        intersect_indices = np.intersect1d(test_idx, val_indices_in_full)

        if len(intersect_indices) == 0:
            continue

        # Map full indices back to df_val relative indices
        val_relative_indices = intersect_indices - val_start_idx

        # Load artifacts for this fold
        try:
            pls = joblib.load(os.path.join(models_dir, f"pls_fold_{fold}.joblib"))
            pls_scaler = joblib.load(
                os.path.join(models_dir, f"pls_scaler_fold_{fold}.joblib")
            )
            tab_scaler = joblib.load(
                os.path.join(models_dir, f"tab_scaler_fold_{fold}.joblib")
            )
            clf = joblib.load(os.path.join(models_dir, f"clf_fold_{fold}.joblib"))
        except FileNotFoundError:
            print(f"Error: Could not find model artifacts for fold {fold}.")
            continue

        # Extract features for these specific samples
        # We can index directly into X_text_full and X_tab_full
        X_text_subset = X_text_full[intersect_indices]
        X_tab_subset = X_tab_full[intersect_indices]

        # Apply pipeline transformations
        X_text_transformed = pls_scaler.transform(pls.transform(X_text_subset))
        X_tab_scaled = tab_scaler.transform(X_tab_subset)
        X_final = np.hstack([X_text_transformed, X_tab_scaled])

        # Predict
        preds = clf.predict_proba(X_final)[:, 1]

        # Store predictions
        val_oof_preds[val_relative_indices] = preds
        val_covered_mask[val_relative_indices] = True

    # Verify we covered all validation samples
    if not np.all(val_covered_mask):
        print("Warning: Not all validation samples were covered by OOF reconstruction.")

    # 4. Metric Calculation
    y_true_val = df_val["requester_received_pizza"].values
    final_auc = roc_auc_score(y_true_val, val_oof_preds)

    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_true_val - val_oof_preds)

    # Create a DataFrame for correlation analysis
    # Use tabular features + error
    analysis_df = pd.DataFrame(X_tab_val, columns=Config.NUMERIC_COLS)
    analysis_df["Error"] = errors

    # Compute correlations
    correlations = (
        analysis_df.corr()["Error"].drop("Error").sort_values(ascending=False, key=abs)
    )

    print("Top features correlated with prediction error:")
    print(correlations.head(5))

    # 6. Submission Logic
    threshold = 0.7141749705260098
    submission_path = Config.SUBMISSION_PATH

    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) > threshold ({threshold}). Keeping submission."
        )
        if os.path.exists(submission_path):
            print(f"Submission file available at: {submission_path}")
        else:
            print("Error: Submission file was not generated by Trainer.")
    else:
        print(
            f"\nValidation metric ({final_auc}) <= threshold ({threshold}). Discarding submission."
        )
        if os.path.exists(submission_path):
            os.remove(submission_path)
            print("Submission file removed.")


if __name__ == "__main__":
    main()
