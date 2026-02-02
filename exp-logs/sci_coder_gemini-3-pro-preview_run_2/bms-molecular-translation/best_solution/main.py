import os
import torch
import pandas as pd
import numpy as np
import scipy.stats
import nltk
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.trainer import Trainer, set_seed
from library.utils import load_checkpoint
from library.dataset import InChiDataset, CollateFn
from library.inference import generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides for Fast Baseline
    # -------------------------------------------------------------------------
    Config.setup()
    set_seed(Config.SEED)

    # Override default configuration for a faster run within 2 hours
    # Using a larger batch size for A100 GPU
    Config.BATCH_SIZE = 128
    # Limit epochs to ensure completion
    Config.EPOCHS = 3
    # Use 8 workers to maximize data loading throughput
    Config.NUM_WORKERS = 8
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print("-" * 40)
    print("Configuration Configured for Fast Baseline")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print("-" * 40)

    # -------------------------------------------------------------------------
    # 2. Data Subsetting
    # -------------------------------------------------------------------------
    # To ensure training finishes quickly, we use a subset of the training data.
    # 50,000 samples is enough to learn the task basics without taking hours.
    print("Creating training subset...")
    full_train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))

    SUBSET_SIZE = 50000
    if len(full_train_df) > SUBSET_SIZE:
        train_subset_df = full_train_df.sample(n=SUBSET_SIZE, random_state=Config.SEED)
    else:
        train_subset_df = full_train_df

    subset_csv_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    train_subset_df.to_csv(subset_csv_path, index=False)
    print(
        f"Training subset saved to {subset_csv_path} ({len(train_subset_df)} samples)"
    )

    # Point Config to the subset
    Config.TRAIN_CSV = subset_csv_path

    # -------------------------------------------------------------------------
    # 3. Training
    # -------------------------------------------------------------------------
    print("\nStarting Training Pipeline...")
    trainer = Trainer(debug=False)
    trainer.fit()

    # -------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model saved during training
    model = trainer.model
    load_checkpoint(Config.MODEL_PATH, model)
    model.eval()

    device = torch.device(Config.DEVICE)
    tokenizer = trainer.tokenizer

    # Use the full validation set as required
    val_dataset = InChiDataset(csv_path=Config.VAL_CSV, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Inference takes less memory
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=CollateFn(),
        pin_memory=(Config.DEVICE == "cuda"),
    )

    lev_distances = []
    img_widths = []
    target_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            target_texts = batch["target_texts"]
            input_lengths = batch["input_lengths"]  # These are the resized widths

            # Forward pass
            logits = model(images)

            # Decode
            preds = tokenizer.decode_ctc_greedy(logits, batch_first=True)

            # Calculate metrics per sample
            for pred, target, width in zip(preds, target_texts, input_lengths):
                dist = nltk.edit_distance(pred, target)
                lev_distances.append(dist)
                img_widths.append(width.item())
                target_lengths.append(len(target))

    # Compute Final Metric
    final_metric = np.mean(lev_distances)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    # We correlate error (Levenshtein distance) with input features
    if len(lev_distances) > 1:
        corr_width, _ = scipy.stats.pearsonr(img_widths, lev_distances)
        corr_len, _ = scipy.stats.pearsonr(target_lengths, lev_distances)

        print("\nFailure Analysis - Error Correlations:")
        print(f"Correlation (Image Width vs Error): {corr_width:.4f}")
        print(f"Correlation (Target Length vs Error): {corr_len:.4f}")

        if abs(corr_len) > 0.5:
            print(
                ">> Strong correlation with sequence length. The model struggles with longer molecules."
            )
        if abs(corr_width) > 0.5:
            print(
                ">> Strong correlation with image width. The model struggles with wider images (resolution loss)."
            )
    else:
        print("Not enough samples for correlation analysis.")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print("\nGenerating Submission for Test Set...")
    generate_submission()  # Generates for full test set using Config.TEST_CSV

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
