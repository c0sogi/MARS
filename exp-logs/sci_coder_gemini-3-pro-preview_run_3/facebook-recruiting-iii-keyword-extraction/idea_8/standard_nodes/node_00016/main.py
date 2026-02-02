import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.data_processing import get_dataloaders
from library.model import WideAndDeepTextCNN, FocalLoss
from library.trainer import ModelTrainer


def main():
    # 1. Setup
    print("Initializing configuration and seeding...")
    Config.setup()
    seed_everything(Config.SEED)

    # 2. Data Loading
    print("Loading data...")
    # load_cached_data=True ensures we use preprocessed .npy files if they exist
    train_loader, val_loader, test_loader, mlb = get_dataloaders(load_cached_data=True)

    # Subsample training data for Fast Baseline requirements
    # The full dataset is ~4.3M samples. We restrict to 500k for speed.
    MAX_TRAIN_SAMPLES = 500000
    if len(train_loader.dataset) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(train_loader.dataset)} to {MAX_TRAIN_SAMPLES} samples..."
        )
        # Use fixed generator for reproducibility in subset selection
        g = torch.Generator()
        g.manual_seed(Config.SEED)
        indices = torch.randperm(len(train_loader.dataset), generator=g)[
            :MAX_TRAIN_SAMPLES
        ]

        train_subset = Subset(train_loader.dataset, indices)

        # Re-create the train loader with the subset
        train_loader = DataLoader(
            train_subset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

    # 3. Model Initialization
    print("Initializing model...")
    model = WideAndDeepTextCNN(
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_tags=Config.NUM_TAGS,
        cnn_filters=Config.CNN_FILTERS,
        cnn_kernel_sizes=Config.CNN_KERNEL_SIZES,
        dropout=Config.DROPOUT,
    )

    # 4. Optimizer, Loss, Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = FocalLoss()

    # OneCycleLR requires steps_per_epoch
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    # 5. Training
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
    )

    print("Starting training...")
    trainer.fit(num_epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE)

    # 6. Validation & Failure Analysis
    print("Loading best model for validation and analysis...")
    checkpoint = load_checkpoint(Config.MODEL_PATH, model, device=Config.DEVICE)
    best_threshold = checkpoint["best_threshold"]

    model.eval()
    val_probs = []
    val_targets = []
    val_lens = []

    # Inference loop on validation set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(Config.DEVICE)

            # Compute sequence length (number of non-pad tokens) for failure analysis
            # Pad token ID is 0
            seq_len = (inputs != 0).sum(dim=1).cpu().numpy()
            val_lens.append(seq_len)

            # Mixed precision inference
            with torch.cuda.amp.autocast():
                logits = model(inputs)

            probs = torch.sigmoid(logits)
            val_probs.append(probs.cpu().numpy())
            val_targets.append(targets.numpy())

    val_probs = np.vstack(val_probs)
    val_targets = np.vstack(val_targets)
    val_lens = np.concatenate(val_lens)

    # Calculate Final Metric
    val_preds = (val_probs >= best_threshold).astype(int)
    final_f1 = f1_score(val_targets, val_preds, average="samples", zero_division=0)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    # Calculate sample-wise F1 to correlate with sequence length
    # Vectorized F1 calculation
    tp = (val_preds * val_targets).sum(axis=1)
    fp = (val_preds * (1 - val_targets)).sum(axis=1)
    fn = ((1 - val_preds) * val_targets).sum(axis=1)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    sample_f1s = 2 * (precision * recall) / (precision + recall + epsilon)

    error_magnitude = 1.0 - sample_f1s

    # Correlation
    correlation = np.corrcoef(val_lens, error_magnitude)[0, 1]
    print(
        f"Failure Analysis: Correlation between Input Sequence Length and Error: {correlation:.4f}"
    )

    # 7. Submission
    THRESHOLD_SCORE = 0.33488
    if final_f1 > THRESHOLD_SCORE:
        print("Metric above threshold. Generating submission...")

        # Predict on Test Set
        test_probs = trainer.predict(test_loader)
        test_preds = (test_probs >= best_threshold).astype(int)

        # Decode Tags
        print("Decoding tags...")
        # mlb.inverse_transform returns a list of tuples of tags
        pred_tags_tuples = mlb.inverse_transform(test_preds)
        pred_tags_strings = [" ".join(tags) for tags in pred_tags_tuples]

        # Load Test IDs from Metadata
        df_test_meta = pd.read_csv(Config.TEST_META_FILE)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"Id": df_test_meta["Id"], "Tags": pred_tags_strings}
        )

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Final metric {final_f1} is not greater than {THRESHOLD_SCORE}. Skipping submission."
        )


if __name__ == "__main__":
    main()
