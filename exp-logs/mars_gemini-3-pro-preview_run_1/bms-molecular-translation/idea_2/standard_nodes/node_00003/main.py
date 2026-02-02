import os
import random
import numpy as np
import torch
import pandas as pd
import cv2
import nltk
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.tokenizer import Tokenizer
from library.model import Seq2Seq
from library.engine import fit, inference
from library.utils import load_checkpoint
from library.dataset import get_loaders


def seed_everything(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_failure_analysis(config, model, tokenizer, val_loader):
    """
    Performs failure analysis on the validation set.
    Calculates Levenshtein distance and correlates it with input features.
    """
    print("\n--- Starting Failure Analysis ---")
    model.eval()

    predictions = []
    ground_truths = []

    # 1. Generate Predictions
    print("Generating predictions on validation set...")
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(config.device)
            preds = model.predict(images)

            decoded_preds = [tokenizer.sequence_to_text(p) for p in preds]
            decoded_labels = [tokenizer.sequence_to_text(l) for l in labels]

            predictions.extend(decoded_preds)
            ground_truths.extend(decoded_labels)

    # 2. Calculate Metrics per sample
    lev_distances = []
    for p, t in zip(predictions, ground_truths):
        lev_distances.append(nltk.edit_distance(p, t))

    # Calculate final metric on full set before subsetting for analysis
    final_metric = np.mean(lev_distances)
    print(f"\nFinal Validation Metric (Full Set): {final_metric}")

    # 3. Gather Metadata (Original Dimensions)
    print("Gathering image metadata for correlation analysis...")

    widths = []
    heights = []
    seq_lens = []

    # Re-load and subset validation dataframe to match the loader's data
    full_val_df = pd.read_csv(config.val_metadata_path)
    if config.debug and config.subset_size:
        current_val_df = full_val_df.iloc[: config.subset_size].copy()
    else:
        current_val_df = full_val_df.copy()

    # Ensure alignment (truncate if necessary, though lengths should match)
    min_len = min(len(current_val_df), len(lev_distances))
    current_val_df = current_val_df.iloc[:min_len]
    lev_distances = lev_distances[:min_len]

    # OPTIMIZATION: Sample for correlation analysis to save time on large datasets
    # Reading 380k images sequentially is too slow.
    analysis_sample_size = 5000
    if len(current_val_df) > analysis_sample_size:
        print(f"Subsetting analysis to {analysis_sample_size} samples for speed...")
        # Use numpy choice for random indices
        indices = np.random.choice(
            len(current_val_df), analysis_sample_size, replace=False
        )
        current_val_df = current_val_df.iloc[indices]
        # Subset lev_distances list using list comprehension or numpy
        lev_distances = [lev_distances[i] for i in indices]

    # Read original image dimensions
    for _, row in current_val_df.iterrows():
        path = os.path.join(config.input_dir, row["file_path"])
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
        else:
            # Fallback for missing images
            widths.append(0)
            heights.append(0)

        seq_lens.append(len(row["InChI"]))

    # 4. Compute Correlations
    metrics_df = pd.DataFrame(
        {
            "error": lev_distances,
            "width": widths,
            "height": heights,
            "seq_len": seq_lens,
        }
    )

    # Filter out invalid reads
    metrics_df = metrics_df[metrics_df["width"] > 0]

    print("\nCorrelations with Error Magnitude (Levenshtein Distance):")
    for feature in ["width", "height", "seq_len"]:
        if metrics_df[feature].std() > 0:
            corr, _ = pearsonr(metrics_df["error"], metrics_df[feature])
            print(f"  Error vs {feature}: {corr:.4f}")
        else:
            print(f"  Error vs {feature}: NaN (No variance)")


def main():
    # 1. Setup
    seed_everything(42)

    # Configure for a full run
    # debug=False ensures the full validation set is used
    config = Config(debug=False)

    # Override defaults to optimize for A100 GPU and 12 vCPUs
    config.epochs = 1  # Train for 1 epoch given time constraints
    config.batch_size = 256  # Increased batch size for A100 efficiency
    config.num_workers = 10  # Increased workers for faster data loading
    config.subset_size = None

    config.print_config()

    # 2. Train
    print("\n=== STEP 1: Training ===")
    fit(config)

    # 3. Validation & Failure Analysis
    print("\n=== STEP 2: Validation & Analysis ===")

    # Load tokenizer
    tokenizer = Tokenizer(config)
    tokenizer.load_or_build_vocab(load_cached_data=True)

    # Get Loaders (specifically to get the validation loader)
    _, val_loader, _ = get_loaders(config, tokenizer)

    # Initialize Model
    model = Seq2Seq(config, vocab_size=len(tokenizer)).to(config.device)

    # Load Best Weights
    if os.path.exists(config.model_save_path):
        load_checkpoint(config.model_save_path, model, device=config.device)
    elif os.path.exists(config.checkpoint_path):
        load_checkpoint(config.checkpoint_path, model, device=config.device)
    else:
        print("Warning: No checkpoint found. Analysis will use random weights.")

    # Run Analysis
    run_failure_analysis(config, model, tokenizer, val_loader)

    # 4. Inference / Submission
    print("\n=== STEP 3: Submission Generation ===")
    inference(config)

    print("\nRun complete.")


if __name__ == "__main__":
    main()
