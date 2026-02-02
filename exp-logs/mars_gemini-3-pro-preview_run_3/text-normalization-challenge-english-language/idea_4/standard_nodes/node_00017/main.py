import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from nltk import edit_distance

# Import from provided libraries
from library.config import Config
from library.dataset_factory import create_dataloaders
from library.trainer import ModelTrainer
from library.inference_pipeline import CascadeSolver
from library.symbolic_layer import SymbolicMemory
from library.retrieval_system import SimilarityIndex


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)

    # Override Config for Fast Baseline
    print("Configuring for Fast Baseline execution...")
    Config.MAX_TRAIN_SAMPLES = 100000  # Limit samples for speed
    Config.EPOCHS = 5  # Limit epochs
    Config.BATCH_SIZE = 128  # Ensure stable batch size

    # Ensure directories exist
    Config.setup()

    # 2. Data Preparation
    print("\n=== Data Preparation ===")
    # This handles Tokenizer training, Index building, and Dataset creation
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        load_cached_data=True
    )

    # 3. Model Training
    print("\n=== Model Training ===")
    trainer = ModelTrainer(tokenizer)
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 4. Validation Assessment
    print("\n=== Validation Assessment ===")
    # Load full validation metadata to evaluate the entire cascade
    if not os.path.exists(Config.VAL_DATA_PATH):
        raise FileNotFoundError(f"Validation data not found at {Config.VAL_DATA_PATH}")

    df_val = pd.read_parquet(Config.VAL_DATA_PATH)

    # Initialize Solver (loads the best model checkpoint we just trained)
    solver = CascadeSolver()

    # Run inference on validation set
    # The solver expects a dataframe with 'before', 'sentence_id', 'token_id', 'id'
    print(f"Running cascade inference on {len(df_val)} validation samples...")
    val_predictions = solver.solve(df_val)

    # Merge predictions with ground truth
    # df_val has 'id' and 'after' (ground truth)
    # val_predictions has 'id' and 'after' (predicted)
    comparison_df = df_val[["id", "before", "after"]].merge(
        val_predictions, on="id", how="left", suffixes=("_true", "_pred")
    )

    # Handle any potential missing predictions (should be handled by solver, but safety first)
    comparison_df["after_pred"] = comparison_df["after_pred"].fillna(
        comparison_df["before"]
    )

    # Calculate Metric
    # Prediction accuracy (total percent of correct tokens)
    # Exact string match required
    correct_predictions = comparison_df["after_true"] == comparison_df["after_pred"]
    accuracy = correct_predictions.mean()

    print(f"Final Validation Metric: {accuracy}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Filter for errors
    errors_df = comparison_df[~correct_predictions].copy()

    if len(errors_df) > 0:
        print(f"Analyzing {len(errors_df)} errors...")

        # Calculate Error Magnitude (Levenshtein Distance)
        # We use nltk.edit_distance
        errors_df["error_magnitude"] = errors_df.apply(
            lambda row: edit_distance(str(row["after_true"]), str(row["after_pred"])),
            axis=1,
        )

        # Calculate Input Feature (Length of 'before' token)
        errors_df["input_length"] = errors_df["before"].astype(str).apply(len)

        # Calculate Correlation
        correlation = errors_df["error_magnitude"].corr(errors_df["input_length"])

        print(
            f"Correlation between Error Magnitude (Levenshtein) and Input Length: {correlation}"
        )

        print("Top 5 Errors (by magnitude):")
        top_errors = errors_df.sort_values("error_magnitude", ascending=False).head(5)
        for _, row in top_errors.iterrows():
            print(
                f"Input: {row['before']} | True: {row['after_true']} | Pred: {row['after_pred']} | Dist: {row['error_magnitude']}"
            )
    else:
        print("No errors found in validation set (Perfect Score).")

    # 6. Submission Generation
    print("\n=== Submission Generation ===")
    THRESHOLD = 0.9943860453286453

    if accuracy > THRESHOLD:
        print(
            f"Validation accuracy ({accuracy}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        if not os.path.exists(Config.TEST_DATA_PATH):
            raise FileNotFoundError(f"Test data not found at {Config.TEST_DATA_PATH}")

        df_test = pd.read_parquet(Config.TEST_DATA_PATH)

        # Run Solver on Test Data
        submission_df = solver.solve(df_test)

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation accuracy ({accuracy}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
