import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data_processing import load_raw_data, get_tfidf_features
from library.tapt_trainer import run_tapt
from library.supervised_trainer import run_supervised_training
from library.models import LinearModelWrapper
from library.ensemble import (
    optimize_blending_weights,
    blend_predictions,
    create_submission,
    get_val_targets,
    get_test_ids,
)


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("=== Initializing Pipeline ===")

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Runtime Optimization Strategy:
    # To fit within the 2-hour limit on the provided hardware:
    # 1. TAPT: Run on a subset of data (40k per split -> ~120k total) for 1 epoch.
    # 2. Fine-Tuning: Run on the full dataset but for 1 epoch.

    Config.DEBUG_SAMPLES = 40000  # Increase debug samples for robust TAPT
    Config.TRAIN_PARAMS["epochs"] = 1  # Limit supervised training to 1 epoch

    # Paths for TAPT artifacts
    tapt_a_path = Config.MODEL_A_TAPT_PATH
    tapt_b_path = Config.MODEL_B_TAPT_PATH

    # =========================================================================
    # 2. Task-Adaptive Pretraining (TAPT)
    # =========================================================================
    print("\n[Stage 1] Running Task-Adaptive Pretraining (TAPT)...")

    # Train Model A (DeBERTa) Backbone
    # Check for the specific artifact (pytorch_model.bin) rather than just the directory
    if not os.path.exists(os.path.join(tapt_a_path, "pytorch_model.bin")):
        run_tapt(
            model_name=Config.MODEL_A_NAME,
            output_path=tapt_a_path,
            debug=True,  # Use subset defined by Config.DEBUG_SAMPLES
            epochs=1,  # 1 epoch is sufficient for domain adaptation
        )
    else:
        print(f"TAPT weights for {Config.MODEL_A_NAME} found, skipping.")

    # Train Model B (RoBERTa) Backbone
    if not os.path.exists(os.path.join(tapt_b_path, "pytorch_model.bin")):
        run_tapt(
            model_name=Config.MODEL_B_NAME,
            output_path=tapt_b_path,
            debug=True,
            epochs=1,
        )
    else:
        print(f"TAPT weights for {Config.MODEL_B_NAME} found, skipping.")

    # =========================================================================
    # 3. Linear Baseline Training
    # =========================================================================
    print("\n[Stage 2] Training Linear Baseline...")

    # Load full data for Linear Model
    train_df, val_df, test_df = load_raw_data(debug=False)

    # Generate/Load TF-IDF Features
    X_train, X_val, X_test = get_tfidf_features(
        train_df["comment_text"],
        val_df["comment_text"],
        test_df["comment_text"],
        load_cached_data=True,
    )

    # Train Logistic Regression Ensemble
    linear_model = LinearModelWrapper()
    y_train = train_df[Config.LABEL_COLS].values
    linear_model.fit(X_train, y_train)

    # Generate Predictions
    val_preds_linear = linear_model.predict_proba(X_val)
    test_preds_linear = linear_model.predict_proba(X_test)

    # Free memory
    del X_train, X_val, X_test, linear_model
    import gc

    gc.collect()

    # =========================================================================
    # 4. Supervised Fine-Tuning
    # =========================================================================
    print("\n[Stage 3] Running Supervised Fine-Tuning...")

    # Model A: DeBERTa (Fine-tune on full data)
    val_preds_a, test_preds_a = run_supervised_training(
        model_name=Config.MODEL_A_NAME,
        pretrained_path=None,  # Use standard pre-trained weights
        save_model_path=Config.MODEL_A_BEST_PATH,
        val_preds_save_path=os.path.join(Config.WORKING_DIR, "val_preds_a.npy"),
        test_preds_save_path=os.path.join(Config.WORKING_DIR, "test_preds_a.npy"),
        debug=False,  # Use full dataset
    )

    # Model B: RoBERTa (Fine-tune on full data)
    val_preds_b, test_preds_b = run_supervised_training(
        model_name=Config.MODEL_B_NAME,
        pretrained_path=None,  # Use standard pre-trained weights
        save_model_path=Config.MODEL_B_BEST_PATH,
        val_preds_save_path=os.path.join(Config.WORKING_DIR, "val_preds_b.npy"),
        test_preds_save_path=os.path.join(Config.WORKING_DIR, "test_preds_b.npy"),
        debug=False,  # Use full dataset
    )

    # =========================================================================
    # 5. Ensemble Optimization & Validation
    # =========================================================================
    print("\n[Stage 4] Ensembling & Validation...")

    # Load targets
    val_targets = get_val_targets()

    # Collect predictions
    preds_list = [val_preds_a, val_preds_b, val_preds_linear]

    # Optimize weights
    weights = optimize_blending_weights(val_targets, preds_list)

    # Compute Final Metric
    blended_val_preds = blend_predictions(preds_list, weights)
    final_score = calculate_roc_auc(val_targets, blended_val_preds)

    print(f"Final Validation Metric: {final_score}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    print("\n[Stage 5] Failure Analysis...")

    # Calculate Mean Absolute Error per sample (averaged across 6 classes)
    # shape: (N_val,)
    error_magnitude = np.abs(val_targets - blended_val_preds).mean(axis=1)

    # Get text lengths from validation dataframe
    # We loaded val_df earlier with debug=False, so it matches val_targets
    text_lengths = val_df["comment_text"].apply(len).values

    # Calculate correlation
    corr, p_value = pearsonr(error_magnitude, text_lengths)
    print(f"Correlation between Error Magnitude and Input Length: {corr:.6f}")

    # =========================================================================
    # 7. Submission Generation
    # =========================================================================
    threshold = 0.9927306969806252

    if final_score > threshold:
        print(
            f"\nValidation score {final_score} exceeds threshold {threshold}. Generating submission..."
        )

        # Blend test predictions using optimized weights
        test_preds_list = [test_preds_a, test_preds_b, test_preds_linear]
        final_test_preds = blend_predictions(test_preds_list, weights)

        # Get Test IDs
        test_ids = get_test_ids()

        # Create submission file
        create_submission(test_ids, final_test_preds)
    else:
        print(
            f"\nValidation score {final_score} does not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
