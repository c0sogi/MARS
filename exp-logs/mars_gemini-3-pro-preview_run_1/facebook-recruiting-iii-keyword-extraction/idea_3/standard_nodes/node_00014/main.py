import torch
import pandas as pd
import numpy as np
import sys
import os
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.utils import set_seed
from library.vocabulary import get_or_build_vocabulary
from library.dataset import StackExchangeDataset, collate_fn_offset
from library.model import DualStreamAttentionDAN
from library.engine import train_model, evaluate, predict_test


def run_failure_analysis(model, val_loader, device, threshold=0.5):
    """
    Analyzes model performance on validation set to find correlations
    between error magnitude and input lengths.
    """
    print("\n--- Starting Failure Analysis ---")
    model.eval()

    data_records = []

    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            title_text = batch["title_text"].to(device)
            title_offsets = batch["title_offsets"].to(device)
            body_text = batch["body_text"].to(device)
            body_offsets = batch["body_offsets"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            logits = model(title_text, title_offsets, body_text, body_offsets)
            probs = torch.sigmoid(logits)

            # Binarize predictions
            preds = (probs > threshold).float()

            # Calculate per-sample F1
            # TP = (preds * targets).sum(dim=1)
            # FP = (preds * (1-targets)).sum(dim=1) -> preds.sum() - TP
            # FN = ((1-preds) * targets).sum(dim=1) -> targets.sum() - TP

            tp = (preds * targets).sum(dim=1)
            fp = preds.sum(dim=1) - tp
            fn = targets.sum(dim=1) - tp

            # F1 = 2*TP / (2*TP + FP + FN)
            epsilon = 1e-7
            f1_scores = (2 * tp) / (2 * tp + fp + fn + epsilon)

            # Error Magnitude
            errors = 1.0 - f1_scores

            # Calculate lengths
            # Title lengths
            # Append total length to offsets to calculate diffs
            total_title_tokens = title_text.size(0)
            t_ends = torch.cat(
                [title_offsets[1:], torch.tensor([total_title_tokens], device=device)]
            )
            title_lens = t_ends - title_offsets

            # Body lengths
            total_body_tokens = body_text.size(0)
            b_ends = torch.cat(
                [body_offsets[1:], torch.tensor([total_body_tokens], device=device)]
            )
            body_lens = b_ends - body_offsets

            # Move to CPU and store
            batch_errors = errors.cpu().numpy()
            batch_t_lens = title_lens.cpu().numpy()
            batch_b_lens = body_lens.cpu().numpy()

            for i in range(len(batch_errors)):
                data_records.append(
                    {
                        "error": batch_errors[i],
                        "title_len": batch_t_lens[i],
                        "body_len": batch_b_lens[i],
                    }
                )

    # Create DataFrame
    df_analysis = pd.DataFrame(data_records)

    # Calculate correlations
    corr_title = df_analysis["error"].corr(df_analysis["title_len"])
    corr_body = df_analysis["error"].corr(df_analysis["body_len"])

    print(f"Correlation between Error and Title Length: {corr_title:.4f}")
    print(f"Correlation between Error and Body Length: {corr_body:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Override Config for Fast Baseline on A100
    Config.BATCH_SIZE = 2048
    Config.NUM_EPOCHS = 3
    print(f"Configured Batch Size: {Config.BATCH_SIZE}")
    print(f"Configured Epochs: {Config.NUM_EPOCHS}")

    # 2. Vocabulary
    vocab = get_or_build_vocabulary(load_cached_data=True)

    # 3. Data Loading
    # We instantiate datasets manually to ensure we use the correct collate_fn and settings
    print("Initializing Datasets...")
    train_dataset = StackExchangeDataset("train", vocab, load_cached_data=True)
    val_dataset = StackExchangeDataset("val", vocab, load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn_offset,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn_offset,
        pin_memory=Config.PIN_MEMORY,
    )

    # 4. Model Initialization
    print("Initializing Model...")
    model = DualStreamAttentionDAN(
        vocab_size=vocab.get_vocab_size(),
        num_classes=vocab.get_num_tags(),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
        init_range=Config.INIT_RANGE,
    )
    model.to(device)

    # 5. Training
    print("Starting Training Loop...")
    train_model(model, train_loader, val_loader, device, num_epochs=Config.NUM_EPOCHS)

    # 6. Final Validation
    print("Performing Final Validation...")
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    val_loss, val_f1 = evaluate(model, val_loader, criterion, device)

    # Required Output Format
    print(f"Final Validation Metric: {val_f1}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Submission
    TARGET_THRESHOLD = 0.6000850796699524
    if val_f1 > TARGET_THRESHOLD:
        print(
            f"Validation F1 ({val_f1}) exceeds threshold ({TARGET_THRESHOLD}). Generating submission..."
        )

        # Create Test Loader
        test_loader = DataLoader(
            StackExchangeDataset("test", vocab, load_cached_data=True),
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn_offset,
            pin_memory=Config.PIN_MEMORY,
        )

        predict_test(vocab, test_loader, device)
    else:
        print(
            f"Validation F1 ({val_f1}) did not exceed threshold ({TARGET_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
