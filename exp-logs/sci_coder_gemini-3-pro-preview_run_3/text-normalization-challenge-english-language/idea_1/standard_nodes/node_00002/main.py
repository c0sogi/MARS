import os
import pandas as pd
import numpy as np
import warnings
from library.utils import set_seed, setup_logger
from library.stats_model import HierarchicalLookupModel
from library.data_loader import load_dataset

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Initialization
    set_seed(42)
    logger = setup_logger("Runfile")

    # Initialize the hierarchical model
    # The model uses a cache directory to store learned statistics
    model = HierarchicalLookupModel(cache_dir="./working/idea_1")

    # 2. Training
    # The fit method calculates L1 (Unigram) and L2 (Bigram) statistics from the training data.
    # It handles data loading and context processing internally.
    logger.info("Starting model training...")
    model.fit(train_split="train", load_cached_data=True)

    # 3. Validation
    # Evaluate the model on the hold-out validation set.
    logger.info("Starting validation...")
    val_accuracy = model.evaluate(val_split="val", load_cached_data=True)

    # REQUIRED: Print the final metric in the specific format
    print(f"Final Validation Metric: {val_accuracy}")

    # 4. Failure Analysis
    logger.info("Performing failure analysis...")

    # Load validation data explicitly to access features
    df_val = load_dataset(split="val", process_context=True, load_cached_data=True)

    # Generate predictions
    preds = model.predict(df_val)
    df_val["pred"] = preds

    # Identify errors (Exact string match required)
    # Ensure both columns are strings to avoid type mismatch errors
    df_val["after"] = df_val["after"].astype(str)
    df_val["pred"] = df_val["pred"].astype(str)
    df_val["is_error"] = (df_val["after"] != df_val["pred"]).astype(int)

    # Extract simple features for correlation analysis
    # We want to see if errors correlate with specific token characteristics
    df_val["len_before"] = df_val["before"].str.len()
    df_val["num_digits"] = df_val["before"].apply(
        lambda x: sum(c.isdigit() for c in str(x))
    )
    df_val["num_alpha"] = df_val["before"].apply(
        lambda x: sum(c.isalpha() for c in str(x))
    )
    df_val["is_upper"] = df_val["before"].apply(lambda x: 1 if str(x).isupper() else 0)
    df_val["is_title"] = df_val["before"].apply(lambda x: 1 if str(x).istitle() else 0)

    # Calculate correlation between Error (0/1) and features
    features = ["len_before", "num_digits", "num_alpha", "is_upper", "is_title"]
    correlations = df_val[features].corrwith(df_val["is_error"])

    print("\n--- Failure Analysis: Correlation with Error Magnitude ---")
    print(correlations)
    print("----------------------------------------------------------\n")

    # 5. Submission
    # Generate predictions for the test set and save to CSV
    baseline_metric = 0.991657606341203
    if val_accuracy > baseline_metric:
        logger.info(
            f"Validation accuracy {val_accuracy} exceeds baseline {baseline_metric}. Generating submission..."
        )
        submission_path = "./submission/submission.csv"
        model.generate_submission(
            test_split="test", output_file=submission_path, load_cached_data=True
        )

        if os.path.exists(submission_path):
            logger.info(f"Submission successfully saved to {submission_path}")
        else:
            logger.error("Submission file was not created.")
    else:
        logger.info(
            f"Validation accuracy {val_accuracy} did not exceed baseline {baseline_metric}. Skipping submission."
        )


if __name__ == "__main__":
    main()
