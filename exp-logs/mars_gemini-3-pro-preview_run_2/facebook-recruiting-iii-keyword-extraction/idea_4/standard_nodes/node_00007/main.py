import sys
import os
import pandas as pd
import numpy as np
import torch
import gc
import shutil

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, Timer
from library.data_loader import get_dataloaders
from library.model import DistilRobertaForTagging
from library.trainer import Trainer
from library.inference import (
    get_predictions,
    optimize_threshold,
    generate_submission,
    load_tag_vocab,
)


def calculate_f1_score(probs, labels, threshold):
    """
    Calculates Mean F1-Score (Samples) given probabilities, labels, and a threshold.
    """
    preds = (probs > threshold).astype(np.uint8)
    labels = labels.astype(np.uint8)

    # F1 (Samples) = 2 * |P intersect T| / (|P| + |T|)
    intersection = (preds * labels).sum(axis=1)
    pred_sum = preds.sum(axis=1)
    true_sum = labels.sum(axis=1)

    denominator = pred_sum + true_sum
    # Avoid division by zero: if both pred and true are empty, intersection is 0, F1 is 1 (correctly predicted empty)
    # But in this task, questions usually have tags. If denom is 0, it means no tags predicted and no tags true.
    # We set safe denominator.
    safe_denom = np.maximum(denominator, 1e-9)

    f1_scores = 2 * intersection / safe_denom
    return np.mean(f1_scores), f1_scores


def main():
    # 1. Setup and Configuration
    set_seed(Config.seed)

    # Modify Config for fast baseline execution
    Config.epochs = 1
    Config.train_batch_size = 128
    Config.valid_batch_size = 256

    # Define paths for subsampled data
    full_train_path = Config.train_path
    subsampled_train_path = os.path.join(Config.working_dir, "train_subsampled.csv")

    print("Preparing training data...")
    # Load full train, subsample 1M, save
    # We use engine='c' for speed
    df_train = pd.read_csv(full_train_path, engine="c")
    print(f"Original train shape: {df_train.shape}")

    # Subsample 1,000,000 samples for training to ensure < 2h runtime
    if len(df_train) > 1000000:
        print("Subsampling to 1,000,000 samples...")
        df_train = df_train.sample(n=1000000, random_state=Config.seed).reset_index(
            drop=True
        )

    print(f"Subsampled train shape: {df_train.shape}")
    df_train.to_csv(subsampled_train_path, index=False)

    # Update Config to point to subsampled data
    Config.train_path = subsampled_train_path

    # Clear cached files to force regeneration for the new subsampled data
    # We must clear labels and tags to ensure consistency
    files_to_remove = [
        os.path.join(Config.working_dir, "train_labels.npy"),
        os.path.join(Config.working_dir, "tags.json"),
        os.path.join(
            Config.working_dir, "val_probs.npy"
        ),  # Clear val cache to ensure we use new model
        os.path.join(Config.working_dir, "test_probs.npy"),  # Clear test cache
    ]

    for fpath in files_to_remove:
        if os.path.exists(fpath):
            os.remove(fpath)

    # Clean up memory
    del df_train
    gc.collect()

    # 2. Load Data
    print("Loading DataLoaders...")
    # This will generate new tags.json based on the subsampled data and create new label matrices
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Initialize Model
    print("Initializing Model...")
    model = DistilRobertaForTagging(Config)

    # 4. Train
    print("Starting Training...")
    trainer = Trainer(model, train_loader, val_loader, Config)
    trainer.fit()

    # 5. Validation Inference & Threshold Optimization
    print("Running Validation Inference...")
    # We pass load_cached_data=False to force prediction using the newly trained model
    # (though we deleted the cache file above, so it would compute anyway)
    val_probs, val_labels, val_ids = get_predictions(
        model, val_loader, "val", load_cached_data=False
    )

    # Optimize threshold
    best_threshold = optimize_threshold(val_probs, val_labels)

    # Calculate Final Metric
    final_metric, f1_per_sample = calculate_f1_score(
        val_probs, val_labels, best_threshold
    )
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load validation metadata to get text features
    df_val = pd.read_csv(Config.val_path, engine="c")

    # Ensure alignment. val_ids contains the IDs in the order of predictions.
    # We create a lookup dictionary or reindex the dataframe.
    df_val.set_index("Id", inplace=True)

    # Select rows matching the validation predictions
    # Note: val_ids might be a subset if debug mode was on, but here we run full.
    # Using .loc ensures we get the rows in the correct order corresponding to val_ids
    aligned_df = df_val.loc[val_ids]

    # Calculate lengths
    # Fillna is important as some bodies might be empty (though rare)
    title_lengths = aligned_df["Title"].fillna("").astype(str).apply(len).values
    body_lengths = aligned_df["Body"].fillna("").astype(str).apply(len).values

    # Error magnitude = 1 - F1
    error_magnitude = 1.0 - f1_per_sample

    # Correlations
    corr_title = np.corrcoef(error_magnitude, title_lengths)[0, 1]
    corr_body = np.corrcoef(error_magnitude, body_lengths)[0, 1]

    print(f"Correlation between Error Magnitude and Title Length: {corr_title:.4f}")
    print(f"Correlation between Error Magnitude and Body Length: {corr_body:.4f}")

    # Clean up
    del df_val, aligned_df, title_lengths, body_lengths
    gc.collect()

    # 7. Submission
    target_metric = 0.0542101508997596
    if final_metric > target_metric:
        print(f"\nMetric {final_metric} > {target_metric}. Generating submission...")

        # Run inference on test set
        test_probs, _, test_ids = get_predictions(
            model, test_loader, "test", load_cached_data=False
        )

        # Load tag vocabulary to decode predictions
        tag_vocab = load_tag_vocab()

        # Generate submission file
        generate_submission(
            test_probs, test_ids, best_threshold, tag_vocab, Config.submission_path
        )
    else:
        print(f"\nMetric {final_metric} <= {target_metric}. Skipping submission.")


if __name__ == "__main__":
    main()
