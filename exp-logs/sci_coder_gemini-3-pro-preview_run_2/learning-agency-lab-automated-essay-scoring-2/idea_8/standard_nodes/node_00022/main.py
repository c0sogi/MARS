import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

# Import necessary components from the provided library
from library.config import CFG
from library.utils import seed_everything, get_logger, get_score, OptimizedRounder
from library.workflow import run_classic_branches, run_nn_fold
from library.data import get_loaders, get_test_loader
from library.classic_models import get_lgbm_stacker


def main():
    # --- 1. Setup & Configuration ---
    seed_everything(CFG.seed)

    # Override Configuration for Fast Baseline Execution
    # Reducing epochs to 1 ensures the run completes well within the 2-hour limit
    # while still allowing the pre-trained transformer to learn task-specific features.
    CFG.epochs = 1
    CFG.train_batch_size = 4
    CFG.valid_batch_size = 8

    # Ensure output directory exists
    os.makedirs(CFG.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(CFG.output_dir, "runfile"))
    logger.info("Starting execution of Idea 8: Heterogeneous Stacking Network")

    # --- 2. Run Classic Branches (Lexical, Morphological, Structural) ---
    logger.info("Step 1: Running Classic Feature Branches...")
    # This computes TF-IDF Ridge models and Structural features
    lex_oof, morph_oof, struct_oof, lex_test, morph_test, struct_test, targets = (
        run_classic_branches(load_cached_data=True)
    )

    # --- 3. Run Deep Semantic Branch (Neural Network) ---
    logger.info("Step 2: Running Deep Semantic Branch (DeBERTa-v3)...")

    nn_oof_full = np.zeros_like(targets, dtype=float)
    nn_test_preds_folds = []

    # Load Test Data (Shared across folds)
    test_loader, essay_ids = get_test_loader(load_cached_data=True)

    # Stratified K-Fold Loop
    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)

    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(targets)), targets.astype(int))
    ):
        logger.info(f"--- Processing Fold {fold}/{CFG.num_folds - 1} ---")

        # Get DataLoaders for this fold
        train_loader, val_loader = get_loaders(fold, load_cached_data=True)

        # Train and Validate
        val_preds, test_preds, _ = run_nn_fold(
            fold,
            train_loader,
            val_loader,
            test_loader,
            CFG.device,
            load_cached_data=True,
        )

        # Store OOF Predictions
        nn_oof_full[val_idx] = val_preds

        # Store Test Predictions
        nn_test_preds_folds.append(test_preds)

    # Average Test Predictions across folds
    nn_test_avg = np.mean(nn_test_preds_folds, axis=0)

    # --- 4. Stacking (Meta-Learner) ---
    logger.info("Step 3: Training Meta-Learner (Stacking)...")

    # Construct Meta-Feature Matrix
    # Columns: [NN_Pred, Lexical_Pred, Morph_Pred, Structural_Features...]
    X_meta = np.column_stack([nn_oof_full, lex_oof, morph_oof, struct_oof])
    X_test_meta = np.column_stack([nn_test_avg, lex_test, morph_test, struct_test])

    # Initialize and Train LightGBM
    lgbm = get_lgbm_stacker()
    # Ensure silent execution
    lgbm.set_params(verbosity=-1)
    lgbm.fit(X_meta, targets)

    # Generate Final OOF Predictions
    final_oof_preds = lgbm.predict(X_meta)

    # Optimize Thresholds (Nelder-Mead)
    rounder = OptimizedRounder()
    rounder.fit(final_oof_preds, targets)
    optimized_coeffs = rounder.coefficients()

    # Apply Thresholds to get Integer Scores
    final_oof_rounded = rounder.predict(final_oof_preds, optimized_coeffs)

    # Calculate Validation Metric
    qwk_score = get_score(targets, final_oof_rounded)

    # --- 5. Validation Output ---
    # REQUIRED: Print the final validation metric
    print(f"Final Validation Metric: {qwk_score}")

    # --- 6. Failure Analysis ---
    logger.info("Step 4: Performing Failure Analysis...")

    # Calculate Absolute Error
    errors = np.abs(targets - final_oof_rounded)

    # Define Feature Names for Correlation Analysis
    # Structural features order matches the dict in features.py
    struct_cols = [
        "word_count",
        "sent_count",
        "char_count",
        "avg_word_len",
        "avg_sent_len",
        "unique_ratio",
        "flesch_kincaid",
        "gunning_fog",
        "oov_count",
        "oov_ratio",
    ]
    feature_names = ["nn_pred", "lex_pred", "morph_pred"] + struct_cols

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(X_meta, columns=feature_names)
    analysis_df["error"] = errors

    # Compute Correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)

    print("\nFailure Analysis: Correlation of Features with Error Magnitude")
    print("-" * 50)
    print(correlations)
    print("-" * 50)

    # --- 7. Submission ---
    THRESHOLD = 0.8307992749024942

    if qwk_score > THRESHOLD:
        logger.info(
            f"Validation Score ({qwk_score}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Predict on Test Set
        final_test_preds = lgbm.predict(X_test_meta)

        # Apply Optimized Thresholds
        final_test_rounded = rounder.predict(final_test_preds)

        # Create Submission File
        submission = pd.DataFrame(
            {"essay_id": essay_ids, "score": final_test_rounded.astype(int)}
        )

        submission.to_csv(CFG.submission_file, index=False)
        print(f"Submission saved to {CFG.submission_file}")
    else:
        logger.warning(
            f"Validation Score ({qwk_score}) <= Threshold ({THRESHOLD}). Submission Skipped."
        )


if __name__ == "__main__":
    main()
