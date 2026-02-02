import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1. Suppress tqdm progress bars from library modules to meet requirements
import tqdm


def noop_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = noop_tqdm

# 2. Import Library Modules
# Ensure current directory is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.data import (
    InChiDataset,
    load_and_process_metadata,
    get_transforms,
    Tokenizer,
    set_seed,
)
from library.model import AttributeConditionedModel
from library.train import train_one_epoch, generate_submission
from library.utils import compute_levenshtein

# 3. Configure for Fast Baseline
# Override Config defaults to ensure completion within time limits
Config.EPOCHS = 1
Config.BATCH_SIZE = 128  # Increased batch size for A100 efficiency
Config.NUM_WORKERS = 4
TRAIN_SUBSET_SIZE = 50000  # Limit training data for speed


def main():
    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    print("Initializing DataLoaders...")

    # Load metadata (cached if available)
    train_df = load_and_process_metadata(
        Config.TRAIN_METADATA_PATH, "train_processed", load_cached_data=True
    )
    val_df = load_and_process_metadata(
        Config.VAL_METADATA_PATH, "val_processed", load_cached_data=True
    )
    test_df = load_and_process_metadata(
        Config.TEST_METADATA_PATH, "test_processed", load_cached_data=True
    )

    # Subset Training Data for Fast Baseline
    if len(train_df) > TRAIN_SUBSET_SIZE:
        print(
            f"Subsetting training data from {len(train_df)} to {TRAIN_SUBSET_SIZE} samples."
        )
        train_df = train_df.sample(
            n=TRAIN_SUBSET_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)

    # Initialize Tokenizer
    tokenizer = Tokenizer()

    # Create Datasets
    train_dataset = InChiDataset(
        train_df, tokenizer, transform=get_transforms("train"), mode="train"
    )
    val_dataset = InChiDataset(
        val_df, tokenizer, transform=get_transforms("val"), mode="val"
    )
    test_dataset = InChiDataset(
        test_df, tokenizer, transform=get_transforms("test"), mode="test"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    # Validation and Test loaders use full dataset as required
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing Model...")
    model = AttributeConditionedModel().to(device)

    # Optimizer & Loss
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion_seq = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    criterion_attr = nn.MSELoss()

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        train_loss, seq_loss, attr_loss = train_one_epoch(
            train_loader, model, criterion_seq, criterion_attr, optimizer, device, epoch
        )
        print(
            f"Train Loss: {train_loss:.4f} (Seq: {seq_loss:.4f}, Attr: {attr_loss:.4f})"
        )

    # Save the trained model
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"Model saved to {Config.BEST_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("Starting Validation and Failure Analysis...")
    model.eval()

    total_levenshtein = 0.0
    count = 0

    # Lists for correlation analysis
    error_magnitudes = []
    seq_lengths = []
    attr_errors = []

    with torch.no_grad():
        for images, target_seqs, target_attrs in val_loader:
            images = images.to(device)
            target_attrs = target_attrs.to(device)

            # 1. Predict Sequences (Greedy Decoding)
            pred_seqs = model.predict(images, device=device)
            pred_seqs_np = pred_seqs.cpu().numpy()
            target_seqs_np = target_seqs.numpy()

            # 2. Predict Attributes (for analysis)
            # Manually run the encoder + head part since predict() doesn't return attrs
            visual_feats = model.encoder(images)
            pred_attrs = model.attribute_head(visual_feats)

            # Process batch
            for j in range(len(images)):
                # Decode text
                pred_text = tokenizer.sequence_to_text(pred_seqs_np[j])
                target_text = tokenizer.sequence_to_text(target_seqs_np[j])

                # Compute Metric
                dist = compute_levenshtein(pred_text, target_text)
                total_levenshtein += dist
                count += 1

                # Collect Analysis Data
                error_magnitudes.append(dist)
                seq_lengths.append(len(target_text))

                # Attribute Error (L1 Sum)
                a_err = torch.abs(pred_attrs[j] - target_attrs[j]).sum().item()
                attr_errors.append(a_err)

    # Compute Final Metric
    final_metric = total_levenshtein / count
    print(f"Final Validation Metric: {final_metric}")

    # Compute Correlations
    if count > 0:
        df_analysis = pd.DataFrame(
            {
                "error": error_magnitudes,
                "seq_len": seq_lengths,
                "attr_error": attr_errors,
            }
        )

        corr_len = df_analysis["error"].corr(df_analysis["seq_len"])
        corr_attr = df_analysis["error"].corr(df_analysis["attr_error"])

        print("-" * 30)
        print("Failure Analysis Correlations:")
        print(f"Error magnitude vs Target Sequence Length: {corr_len:.4f}")
        print(f"Error magnitude vs Attribute Prediction Error: {corr_attr:.4f}")
        print("-" * 30)

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    generate_submission(test_loader, model, tokenizer, device)


if __name__ == "__main__":
    main()
