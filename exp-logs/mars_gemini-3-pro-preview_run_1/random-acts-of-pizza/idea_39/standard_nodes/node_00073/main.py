import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

# Import library modules
import library.config
import library.utils
import library.model_tree
import library.model_nn
from library.trainer import Trainer
from library.utils import save_submission

# ==========================================
# 1. Configuration Overrides for Fast Baseline
# ==========================================
# Limit training steps and complexity for a quick execution as required.
library.config.RF_ESTIMATORS = 100
library.config.EPOCHS = 15
library.config.PATIENCE = 5
library.config.DEBUG_MODE = False

# ==========================================
# 2. Monkey Patching to Capture Validation Predictions
# ==========================================
# We need raw validation predictions to compute the ensemble metric and perform failure analysis.
# The provided library functions compute AUC but don't return the prediction arrays.
# We hook into the compute_auc function to capture these arrays during execution.

val_data_storage = {"rf": None, "mlp": []}

# Save reference to the original function
original_compute_auc = library.utils.compute_auc


def hooked_compute_auc(y_true, y_pred):
    """
    Wrapper around compute_auc to intercept and store validation predictions.
    """
    score = original_compute_auc(y_true, y_pred)

    # Store the data.
    # Logic: RF runs first and calls this once. MLP runs second and calls this every epoch.
    entry = {"y_true": y_true, "y_pred": y_pred, "score": score}

    if val_data_storage["rf"] is None:
        val_data_storage["rf"] = entry
    else:
        val_data_storage["mlp"].append(entry)

    return score


# Apply the patch to the modules that import compute_auc
library.model_tree.compute_auc = hooked_compute_auc
library.model_nn.compute_auc = hooked_compute_auc


# ==========================================
# 3. Main Pipeline
# ==========================================
def run_pipeline():
    # Initialize Trainer
    trainer = Trainer()

    # --- Train Random Forest (Stream A) ---
    print("Training Random Forest...")
    rf_ids, rf_test_preds, rf_val_auc = trainer.train_rf(load_cached_data=True)

    # --- Train MLP (Stream B) ---
    print("Training MLP...")
    mlp_ids, mlp_test_preds, mlp_val_auc = trainer.train_mlp(load_cached_data=True)

    # --- Retrieve and Align Validation Predictions ---
    # 1. Random Forest Predictions
    if val_data_storage["rf"] is None:
        raise RuntimeError("Random Forest validation data was not captured.")

    rf_val_entry = val_data_storage["rf"]
    y_val_true = np.array(rf_val_entry["y_true"])
    rf_val_preds = np.array(rf_val_entry["y_pred"])

    # 2. MLP Predictions
    # We need the predictions corresponding to the best AUC reported by the trainer.
    # The trainer returns the best validation AUC found during training.
    best_mlp_entry = None

    # Search for the entry with the matching score (using loose tolerance for float comparison)
    for entry in val_data_storage["mlp"]:
        if np.isclose(entry["score"], mlp_val_auc, atol=1e-7):
            best_mlp_entry = entry
            break

    # Fallback: if exact match not found (unlikely), take the one with the highest score
    if best_mlp_entry is None and val_data_storage["mlp"]:
        best_mlp_entry = max(val_data_storage["mlp"], key=lambda x: x["score"])

    if best_mlp_entry is None:
        raise RuntimeError("MLP validation data was not captured.")

    mlp_val_preds = np.array(best_mlp_entry["y_pred"])

    # Ensure lengths match (handling potential edge cases in data loading)
    if len(rf_val_preds) != len(mlp_val_preds):
        min_len = min(len(rf_val_preds), len(mlp_val_preds))
        rf_val_preds = rf_val_preds[:min_len]
        mlp_val_preds = mlp_val_preds[:min_len]
        y_val_true = y_val_true[:min_len]

    # --- Compute Ensemble Validation Metric ---
    w_rf = library.config.RF_WEIGHT
    w_mlp = library.config.MLP_WEIGHT

    ensemble_val_preds = (rf_val_preds * w_rf) + (mlp_val_preds * w_mlp)
    final_val_auc = roc_auc_score(y_val_true, ensemble_val_preds)

    # PRINT FINAL METRIC AS REQUIRED
    print(f"Final Validation Metric: {final_val_auc}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")
    try:
        # Load validation metadata
        val_df = pd.read_csv(library.config.VAL_PATH)

        # Calculate absolute error
        # Target is binary (0/1), Pred is probability. Error is |y - p|
        errors = np.abs(y_val_true - ensemble_val_preds)

        # Select numeric columns for correlation
        numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [
            "requester_received_pizza",
            "request_id",
            "unix_timestamp_of_request",
            "unix_timestamp_of_request_utc",
        ]
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        correlations = {}
        # Align dataframes if necessary (assuming order is preserved from metadata generation)
        if len(val_df) == len(errors):
            for col in numeric_cols:
                # Fill NaNs with 0 for correlation check
                feat_vals = val_df[col].fillna(0).values
                # Compute correlation
                if np.std(feat_vals) > 0 and np.std(errors) > 0:
                    corr = np.corrcoef(feat_vals, errors)[0, 1]
                    if not np.isnan(corr):
                        correlations[col] = corr

        # Print top correlations
        sorted_corr = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )
        print("Top 5 Features Correlated with Prediction Error:")
        for name, val in sorted_corr[:5]:
            print(f"  {name}: {val:.4f}")

    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # --- Submission Generation ---
    threshold = 0.7056961514236341

    if final_val_auc > threshold:
        print(
            f"\nValidation metric ({final_val_auc}) > threshold ({threshold}). Generating submission."
        )

        # Check ID alignment
        if not np.array_equal(rf_ids, mlp_ids):
            print("Error: Test IDs do not match between models.")
        else:
            # Ensemble Test Predictions
            ensemble_test_preds = trainer.predict_ensemble(
                rf_test_preds, mlp_test_preds
            )

            # Save
            save_submission(rf_ids, ensemble_test_preds)
    else:
        print(
            f"\nValidation metric ({final_val_auc}) <= threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
