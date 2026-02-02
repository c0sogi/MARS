import os
import sys
import numpy as np
import pandas as pd
import torch

# Import necessary modules from the provided library
from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.features import FeatureEngineer
from library.stage1_trainer import Stage1Trainer
from library.stage2_model import LGBMHandler
from library.data import get_data


def main():
    # ------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------
    # Set seed for reproducibility across all operations
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline Execution
    # Reducing epochs to 1 ensures the fine-tuning completes quickly on the A100
    # while still adapting the backbone to the scoring task.
    # Config.EPOCHS = 1

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print("=== Starting Essay Scoring Pipeline ===")

    # ------------------------------------------------------------------
    # 2. Meta-Feature Engineering
    # ------------------------------------------------------------------
    print("\n[Step 1] Generating Meta-Features...")
    # Extracts structural features (lengths, counts) for Train, Val, and Test
    fe = FeatureEngineer()
    fe.run(load_cached_data=False)

    # ------------------------------------------------------------------
    # 3. Stage 1: DeBERTa Fine-Tuning & Embedding Extraction
    # ------------------------------------------------------------------
    print("\n[Step 2] Stage 1 - DeBERTa Training & Extraction...")
    trainer = Stage1Trainer()

    # Train the backbone model (DeBERTa-v3-base)
    # This adapts the generic language model to the specific essay scoring rubric.
    trainer.train_deberta()

    # Extract embeddings using the fine-tuned model
    # We set load_cached_data=False to ensure we generate fresh embeddings
    # from the model we just trained, rather than loading potentially stale files.
    trainer.extract_embeddings(load_cached_data=False)

    # ------------------------------------------------------------------
    # 4. Stage 2: LightGBM Training
    # ------------------------------------------------------------------
    print("\n[Step 3] Stage 2 - LightGBM Training...")
    lgbm_handler = LGBMHandler()

    # Train the gradient boosting model on concatenated features (Embeddings + Meta)
    lgbm_handler.train_model()

    # ------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # ------------------------------------------------------------------
    print("\n[Step 4] Validation and Failure Analysis...")

    # Manually load validation data to compute metrics and perform analysis
    # We need to reconstruct the feature matrix X_val and target vector y_val
    val_emb = np.load(Config.VAL_EMBEDDINGS_PATH)
    val_meta = pd.read_parquet(Config.VAL_META_FEATS_PATH)
    val_df = get_data(Config.VAL_DATA_PATH)

    # Combine dense embeddings and scalar meta-features
    X_val = np.hstack([val_emb, val_meta.values])
    y_val = val_df["score"].values

    # Predict using the trained LightGBM model
    # The model is already trained and stored in lgbm_handler.model
    val_preds_raw = lgbm_handler.model.predict(X_val)

    # Compute Quadratic Weighted Kappa (QWK)
    # compute_qwk handles rounding and clipping internally
    qwk = compute_qwk(y_val, val_preds_raw)

    # Print the Final Validation Metric in the strictly required format
    print(f"Final Validation Metric: {qwk}")

    # --- Failure Analysis ---
    print("\n=== Failure Analysis ===")
    # Calculate absolute error magnitude
    errors = np.abs(y_val - val_preds_raw)

    # Create a temporary dataframe to analyze correlations
    analysis_df = val_meta.copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlation between input meta-features and the error magnitude
    # This helps identify if the model struggles with specific essay characteristics (e.g., short essays)
    correlations = (
        analysis_df.corr()["error_magnitude"]
        .drop("error_magnitude")
        .sort_values(ascending=False)
    )

    print("Correlation between Input Features and Error Magnitude:")
    print(correlations)

    # ------------------------------------------------------------------
    # 6. Submission Generation
    # ------------------------------------------------------------------
    THRESHOLD = 0.8174385126572309

    if qwk > THRESHOLD:
        print(
            f"\nValidation metric ({qwk}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        lgbm_handler.predict_and_submit()
    else:
        print(
            f"\nValidation metric ({qwk}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
