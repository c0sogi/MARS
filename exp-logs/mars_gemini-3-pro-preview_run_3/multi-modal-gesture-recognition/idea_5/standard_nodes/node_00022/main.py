import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.stats
import nltk
from torch.utils.data import DataLoader, Subset

# Import provided library modules
from library.config import Config
from library.trainer import Trainer
from library.dataset import get_datasets
from library.utils import decode_predictions


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline
    Config.NUM_EPOCHS = 8
    Config.BATCH_SIZE = 64  # Increased batch size for A100 efficiency

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    # Ensure directories exist (handled by Trainer init, but good practice)
    Config.setup_directories()

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading datasets...")
    # Load datasets using cached data if available
    train_dataset, val_dataset, test_dataset = get_datasets(load_cached_data=True)

    # Limit training samples for fast baseline requirement
    # We take a random subset of up to 5000 windows if the dataset is larger
    max_train_samples = 5000
    if len(train_dataset) > max_train_samples:
        indices = np.random.choice(len(train_dataset), max_train_samples, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"Training data subsetted to {max_train_samples} samples.")

    # Create DataLoaders
    # num_workers=2 is generally safe and faster than 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # Validation loader (batch_size=1 for full sequence inference)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # ==========================================
    # 3. Model Training
    # ==========================================
    trainer = Trainer()

    # Load Ground Truth for Validation
    val_gt_map = trainer.load_ground_truth(Config.VAL_CSV)

    print(f"Starting training on {trainer.device}...")
    best_score = float("inf")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Run one epoch of training
        train_metrics = trainer.train_epoch(train_loader)

        # Run validation
        val_score = trainer.validate(val_loader, val_gt_map)

        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Levenshtein: {val_score:.4f}"
        )

        # Save best model
        if val_score < best_score:
            best_score = val_score
            torch.save(trainer.model.state_dict(), Config.MODEL_SAVE_PATH)

    # ==========================================
    # 4. Final Validation & Metric
    # ==========================================
    # Load the best model weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=trainer.device)
        )

    # Compute final metric on the full validation set
    final_val_score = trainer.validate(val_loader, val_gt_map)
    print(f"Final Validation Metric: {final_val_score}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("Performing failure analysis...")
    trainer.model.eval()

    lev_distances = []
    seq_lengths = []

    with torch.no_grad():
        for static_x, dynamic_x, sample_ids in val_loader:
            static_x = static_x.to(trainer.device)
            dynamic_x = dynamic_x.to(trainer.device)

            # Forward pass
            _, stage2_logits = trainer.model(static_x, dynamic_x)
            probs = torch.softmax(stage2_logits, dim=2)

            for i in range(len(sample_ids)):
                sid = sample_ids[i]

                # Decode predictions
                pred_seq = decode_predictions(probs[i].cpu().numpy())
                gt_seq = val_gt_map.get(sid, [])

                # Compute Levenshtein distance for this sample
                dist = nltk.edit_distance(pred_seq, gt_seq)

                lev_distances.append(dist)
                # Use sequence length (time dimension) as the feature
                seq_lengths.append(static_x.shape[1])

    # Calculate correlation
    if len(lev_distances) > 1:
        corr, p_val = scipy.stats.pearsonr(lev_distances, seq_lengths)
        print(f"Correlation between Error (Levenshtein) and Sequence Length: {corr}")
    else:
        print("Not enough samples for correlation analysis.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    threshold = 0.30627871362940273

    if final_val_score < threshold:
        print(
            f"Validation score ({final_val_score}) meets threshold ({threshold}). Generating submission..."
        )
        # Trainer.predict() handles loading test data, loading the model, and saving to CSV
        trainer.predict()
    else:
        print(
            f"Validation score ({final_val_score}) is not lower than threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
