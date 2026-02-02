import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

from library.config import Config
from library.utils import set_seed, calculate_f1_score
from library.tokenizer import TextProcessor
from library.dataset import StackExchangeDataset
from library.model import FastTextClassifier
from library.trainer import Trainer


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Tokenizer
    print("Initializing Tokenizer...")
    tokenizer = TextProcessor()
    tokenizer.fit(load_cached_data=True)

    # 3. Data Loading
    print("Preparing Datasets...")
    train_dataset = StackExchangeDataset(
        metadata_path=Config.TRAIN_METADATA,
        tokenizer=tokenizer,
        split_name="train",
        load_cached_data=True,
    )

    val_dataset = StackExchangeDataset(
        metadata_path=Config.VAL_METADATA,
        tokenizer=tokenizer,
        split_name="val",
        load_cached_data=True,
    )

    # Cite solution_lesson_node_00010: Use full dataset for averaging architectures
    # We remove subsampling to maximize data volume.
    print(f"Training on full dataset: {len(train_dataset)} samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=StackExchangeDataset.collate_fn,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=StackExchangeDataset.collate_fn,
        pin_memory=(device == "cuda"),
    )

    # 4. Model Initialization
    vocab_size = tokenizer.get_vocab_size()
    num_classes = tokenizer.get_num_tags()

    print(
        f"Model Config - Vocab: {vocab_size}, Classes: {num_classes}, Embed: {Config.EMBEDDING_DIM}"
    )

    # Cite solution_lesson_node_00005: Upgrade to Deep Averaging Network (DAN)
    model = FastTextClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # 5. Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        device=device,
    )

    print("Starting Training...")
    trainer.fit(epochs=Config.EPOCHS, patience=Config.PATIENCE)

    # 6. Final Validation & Metrics
    print("Performing Final Validation...")
    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    model.eval()
    all_sample_f1s = []
    all_lengths = []

    with torch.no_grad():
        for (
            title_text,
            title_offsets,
            body_text,
            body_offsets,
            targets,
            _,
        ) in val_loader:
            title_text = title_text.to(device)
            title_offsets = title_offsets.to(device)
            body_text = body_text.to(device)
            body_offsets = body_offsets.to(device)
            targets = targets.to(device)

            logits = model(title_text, title_offsets, body_text, body_offsets)
            probs = torch.sigmoid(logits)
            preds = (probs > Config.TAG_THRESHOLD).float()

            # Calculate per-sample F1 manually
            p = preds.cpu().numpy()
            t = targets.cpu().numpy()

            intersection = (p * t).sum(axis=1)
            union_sum = p.sum(axis=1) + t.sum(axis=1)

            batch_f1s = np.divide(
                2 * intersection,
                union_sum,
                out=np.zeros_like(intersection),
                where=union_sum != 0,
            )
            all_sample_f1s.append(batch_f1s)

            # Calculate sequence lengths (use combined length for analysis)
            # Title lengths
            t_offsets = title_offsets.cpu().numpy()
            t_total = title_text.size(0)
            t_extended = np.append(t_offsets, t_total)
            t_lens = t_extended[1:] - t_extended[:-1]

            # Body lengths
            b_offsets = body_offsets.cpu().numpy()
            b_total = body_text.size(0)
            b_extended = np.append(b_offsets, b_total)
            b_lens = b_extended[1:] - b_extended[:-1]

            all_lengths.append(t_lens + b_lens)

    all_sample_f1s = np.concatenate(all_sample_f1s)
    all_lengths = np.concatenate(all_lengths)

    # Calculate Final Metric
    final_f1 = np.mean(all_sample_f1s)
    print(f"Final Validation Metric: {final_f1}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")

    # Error magnitude = 1 - F1
    errors = 1.0 - all_sample_f1s

    # Correlation
    corr, p_val = pearsonr(all_lengths, errors)
    print(
        f"Correlation between Input Sequence Length and Error Magnitude (1-F1): {corr:.4f} (p={p_val:.4e})"
    )

    # 8. Submission
    # Only generate submission if metric meets the threshold
    TARGET_METRIC = 0.6000850796699524
    if final_f1 > TARGET_METRIC:
        print(f"Metric {final_f1} > {TARGET_METRIC}. Generating Submission...")
        test_dataset = StackExchangeDataset(
            metadata_path=Config.TEST_METADATA,
            tokenizer=tokenizer,
            split_name="test",
            load_cached_data=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=StackExchangeDataset.collate_fn,
            pin_memory=(device == "cuda"),
        )

        trainer.generate_submission(test_loader)
    else:
        print(f"Metric {final_f1} <= {TARGET_METRIC}. Skipping Submission.")

    print("Workflow Completed.")


if __name__ == "__main__":
    main()
