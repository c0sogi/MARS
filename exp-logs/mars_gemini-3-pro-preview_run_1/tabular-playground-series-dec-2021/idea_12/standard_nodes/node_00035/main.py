import os
import gc
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.features import preprocess_data
from library.model import XGBoostTrainer


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("runfile")
    logger.info("Starting Optimized Seed Averaging Run...")

    # 2. Load Data
    # Load full data (Cite solution_lesson_node_00002)
    train_df, val_df, test_df = preprocess_data(load_cached_data=True, debug=False)

    logger.info(f"Training Shape: {train_df.shape}")
    logger.info(f"Validation Shape: {val_df.shape}")

    # 3. Target Encoding
    le = LabelEncoder()
    train_df[Config.TARGET_COL] = le.fit_transform(train_df[Config.TARGET_COL])
    val_df[Config.TARGET_COL] = le.transform(val_df[Config.TARGET_COL])
    num_classes = len(le.classes_)
    logger.info(f"Encoded Target. Classes: {num_classes}")

    # 4. Feature Selection
    # Exclude ID and Target
    ignore_cols = [Config.ID_COL, Config.TARGET_COL]
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    logger.info(f"Using {len(feature_cols)} features.")

    # Prepare matrices
    X_train = train_df[feature_cols]
    y_train = train_df[Config.TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[Config.TARGET_COL]
    X_test = test_df[feature_cols]

    # 5. Seed Averaging Loop
    # Train multiple models with different seeds and average predictions (Cite solution_lesson_node_00012)
    SEEDS = [42, 43, 44]
    val_probs_sum = np.zeros((len(val_df), num_classes), dtype=np.float32)
    test_probs_sum = np.zeros((len(test_df), num_classes), dtype=np.float32)

    for seed in SEEDS:
        logger.info(f"\nTraining Model with Seed {seed}...")

        # Initialize Trainer with specific seed
        trainer = XGBoostTrainer(num_class=num_classes)
        trainer.params["random_state"] = seed

        # Fit Model
        model = trainer.fit(X_train, y_train, X_val, y_val)

        # Predict
        val_probs_sum += trainer.predict_proba(model, X_val)
        test_probs_sum += trainer.predict_proba(model, X_test)

        # Cleanup
        del model, trainer
        gc.collect()

    # 6. Ensemble Aggregation
    val_probs_avg = val_probs_sum / len(SEEDS)
    val_preds = np.argmax(val_probs_avg, axis=1)

    test_probs_avg = test_probs_sum / len(SEEDS)
    test_preds = np.argmax(test_probs_avg, axis=1)

    # 7. Metric Calculation
    acc = accuracy_score(y_val, val_preds)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {acc}")

    # 8. Failure Analysis (Continuous Error Magnitude)
    logger.info("Performing Failure Analysis...")
    # Calculate Error Magnitude: 1.0 - Probability of the True Class (Cite solution_lesson_node_00022)
    # Extract prob of true class for each sample
    true_class_probs = val_probs_avg[np.arange(len(y_val)), y_val]
    error_magnitude = 1.0 - true_class_probs

    # Create analysis dataframe
    analysis_df = X_val.copy()
    analysis_df["Error_Magnitude"] = error_magnitude

    # Calculate correlation of features with Error_Magnitude
    corrs = (
        analysis_df.corrwith(analysis_df["Error_Magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    print("\nTop Features Correlated with Error Magnitude:")
    print(corrs.head(6))

    # 9. Submission
    THRESHOLD = 0.9628180555555556

    if acc > THRESHOLD:
        logger.info(
            f"Validation accuracy {acc} > {THRESHOLD}. Generating submission..."
        )

        # Inverse transform labels
        final_preds = le.inverse_transform(test_preds)

        # Create submission DataFrame
        submission = pd.DataFrame(
            {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: final_preds}
        )

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        logger.info(f"Validation accuracy {acc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
