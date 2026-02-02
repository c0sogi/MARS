import os
import random
import numpy as np
import pandas as pd
import torch
import nltk
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import load_checkpoint
from library.tokenizer import Tokenizer
from library.dataset import InChiDataset
from library.model import Image2Seq
from library.trainer import Trainer


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    print("Initializing Fast Baseline Pipeline...")
    set_seed(42)
    config = Config(debug=False)

    # Modify configuration for a fast baseline execution
    config.epochs = 1  # Train for only 1 epoch to save time

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Preparation (Subset for Speed)
    # -------------------------------------------------------------------------
    print("Preparing data...")
    full_train_df = pd.read_csv(config.train_metadata_path)

    # Initialize Tokenizer on FULL data first to ensure complete vocabulary
    # This creates/updates the cached tokenizer_vocab.npy
    tokenizer = Tokenizer(config)
    tokenizer.fit_on_texts(full_train_df["InChI"].values, load_cached_data=True)

    # Create a subset of training data (e.g., 50,000 samples)
    # This ensures the training phase completes well within the time limit
    subset_size = 50000
    if len(full_train_df) > subset_size:
        print(f"Subsetting training data to {subset_size} samples for fast baseline.")
        train_subset_df = full_train_df.sample(n=subset_size, random_state=42)
    else:
        train_subset_df = full_train_df

    # Save subset metadata
    subset_train_path = os.path.join(config.working_dir, "train_subset.csv")
    train_subset_df.to_csv(subset_train_path, index=False)

    # Update config to use the subset
    config.train_metadata_path = subset_train_path

    # -------------------------------------------------------------------------
    # 3. Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Starting Training Phase")
    print("=" * 40)
    trainer = Trainer(config)
    # trainer.fit() handles setup, training loop, and saving best_model.pth
    trainer.fit()

    # -------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Starting Validation & Failure Analysis")
    print("=" * 40)

    # Load full validation metadata
    val_df = pd.read_csv(config.val_metadata_path)
    print(f"Validation set size: {len(val_df)}")

    # Setup Validation Dataset and Loader
    val_dataset = InChiDataset(val_df, tokenizer, config, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Load the best model trained
    model = Image2Seq(config, tokenizer.vocab_size)
    model.to(config.device)
    load_checkpoint(config.best_model_path, model, device=config.device)
    model.eval()

    predictions = []
    ground_truths = []
    target_lengths = []
    distances = []

    print("Running inference on full validation set...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(config.device)

            # Predict (Greedy Decoding)
            batch_preds = model.predict(images, tokenizer)

            # Decode ground truth labels
            batch_targets = []
            for label_seq in labels:
                target_str = tokenizer.sequence_to_text(label_seq)
                batch_targets.append(target_str)
                target_lengths.append(len(target_str))

            # Calculate Levenshtein distance for this batch
            for pred, target in zip(batch_preds, batch_targets):
                d = nltk.edit_distance(pred, target)
                distances.append(d)
                predictions.append(pred)
                ground_truths.append(target)

            if i % 100 == 0:
                print(f"Processed batch {i}/{len(val_loader)}")

    # Compute Final Metric
    final_metric = np.mean(distances)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    # We analyze if longer chemical formulas result in higher errors
    corr, p_value = pearsonr(target_lengths, distances)
    print(f"Correlation (Error Magnitude vs Target Length): {corr:.4f}")

    if abs(corr) > 0.3:
        print(
            "Analysis: Significant correlation detected. Model struggles with longer sequences."
        )
    else:
        print(
            "Analysis: Weak correlation. Errors are likely distributed across sequence lengths."
        )

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Generating Submission")
    print("=" * 40)

    # Trainer.predict handles loading test metadata, inference, and saving submission.csv
    trainer.predict()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
