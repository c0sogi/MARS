import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Add the current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, load_checkpoint, compute_levenshtein
from library.tokenizer import Tokenizer
from library.dataset import InChiDataset, get_transforms
from library.model import Seq2Seq
from library.trainer import Trainer, generate_submission


def main():
    # --- 1. Setup & Configuration ---
    print("--- Starting Runfile Execution ---")
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    # We limit data size and epochs to ensure completion within 2 hours.
    # A100 is fast, but 1.5M images is a lot. We use a representative subset.
    TRAIN_SAMPLE_SIZE = 80000
    VAL_SUBSET_SIZE = 5000
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 128  # A100 can handle this easily

    print(f"Device: {Config.DEVICE}")
    print(f"Training on {TRAIN_SAMPLE_SIZE} samples for {Config.EPOCHS} epochs.")

    # --- 2. Data Preparation ---
    print("\n--- Initializing Tokenizer ---")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(load_cached_data=True)

    print("\n--- Loading Metadata ---")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Subsample for fast training baseline
    train_subset = train_df.sample(
        n=min(len(train_df), TRAIN_SAMPLE_SIZE), random_state=Config.SEED
    ).reset_index(drop=True)
    val_subset = val_df.sample(
        n=min(len(val_df), VAL_SUBSET_SIZE), random_state=Config.SEED
    ).reset_index(drop=True)

    print(f"Train subset shape: {train_subset.shape}")
    print(f"Val subset shape: {val_subset.shape}")

    # Create Datasets
    train_dataset = InChiDataset(
        train_subset, tokenizer, transform=get_transforms("train"), mode="train"
    )
    val_subset_dataset = InChiDataset(
        val_subset, tokenizer, transform=get_transforms("valid"), mode="valid"
    )
    # Full validation dataset for final metric
    val_full_dataset = InChiDataset(
        val_df, tokenizer, transform=get_transforms("valid"), mode="valid"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_subset_loader = DataLoader(
        val_subset_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_full_loader = DataLoader(
        val_full_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Larger batch for inference
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 3. Model Initialization ---
    print("\n--- Initializing Model ---")
    model = Seq2Seq(tokenizer_len=len(tokenizer))
    model.to(Config.DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # --- 4. Training ---
    print("\n--- Starting Training ---")
    trainer = Trainer(model, tokenizer, optimizer, scheduler, device=Config.DEVICE)
    trainer.fit(train_loader, val_subset_loader, epochs=Config.EPOCHS)

    # --- 5. Full Validation Assessment ---
    print("\n--- Executing Full Validation Assessment ---")
    # Load best model
    best_epoch, best_metric = load_checkpoint(
        model, path=Config.MODEL_PATH, device=Config.DEVICE
    )
    model.eval()

    predictions = []
    ground_truths = []
    inchi_lengths = []

    print(f"Running inference on {len(val_full_dataset)} validation samples...")
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(val_full_loader):
            images = images.to(Config.DEVICE)

            # Predict
            batch_preds = model.predict(images, tokenizer)
            predictions.extend(batch_preds)

            # Decode ground truth
            for label in labels:
                text = tokenizer.sequence_to_text(label)
                ground_truths.append(text)
                inchi_lengths.append(len(text))

            if i % 100 == 0:
                print(f"Validated batch {i}/{len(val_full_loader)}")

    # Compute Final Metric
    final_metric = compute_levenshtein(predictions, ground_truths)
    print(f"Final Validation Metric: {final_metric}")

    # --- 6. Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Calculate per-sample Levenshtein distance
    import nltk

    errors = []
    for pred, truth in zip(predictions, ground_truths):
        errors.append(nltk.metrics.distance.edit_distance(pred, truth))

    # Correlation between Error and InChI Length
    if len(errors) > 1 and len(inchi_lengths) > 1:
        corr, _ = pearsonr(errors, inchi_lengths)
        print(f"Correlation (Error vs InChI Length): {corr:.4f}")

        if corr > 0.3:
            print(
                "Analysis: Strong positive correlation. The model struggles more with longer, more complex molecules."
            )
        elif corr < -0.3:
            print(
                "Analysis: Negative correlation. The model struggles with shorter sequences (unlikely)."
            )
        else:
            print(
                "Analysis: Weak correlation. Error is relatively independent of sequence length."
            )

    print(f"Mean Error: {np.mean(errors):.4f}")
    print(f"Max Error: {np.max(errors)}")

    # --- 7. Submission Generation ---
    print("\n--- Generating Submission ---")

    test_dataset = InChiDataset(
        test_df, tokenizer, transform=get_transforms("test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    generate_submission(
        model, test_loader, tokenizer, output_path=Config.SUBMISSION_PATH
    )
    print("Runfile execution complete.")


if __name__ == "__main__":
    main()
