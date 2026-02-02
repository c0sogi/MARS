import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy.stats
import sys
import os

from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.dataset import GestureDataset
from library.trainer import Trainer
from library.postprocessing import (
    generate_submission,
    decode_predictions,
    apply_median_filter,
)


def analyze_failures(trainer, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Computes correlation between error magnitude (Levenshtein distance) and sequence length.
    """
    trainer.model.eval()

    errors = []
    lengths_list = []

    print("\nPerforming Failure Analysis...")

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            seq_labels = batch["seq_labels"]
            batch_lengths = batch["lengths"]

            # Forward pass
            outputs = trainer.model(features, mask)
            stage3_probs = outputs["stage3_cls"]

            for i in range(features.size(0)):
                length = batch_lengths[i].item()
                lengths_list.append(length)

                # Get probability sequence
                probs = stage3_probs[i, :length, :].cpu().numpy()

                # Predict
                pred_labels = np.argmax(probs, axis=1)
                smoothed_labels = apply_median_filter(pred_labels, kernel_size=15)
                pred_sequence = decode_predictions(smoothed_labels)

                # Ground Truth
                gt_sequence = seq_labels[i]
                if isinstance(gt_sequence, torch.Tensor):
                    gt_sequence = gt_sequence.tolist()
                elif isinstance(gt_sequence, np.ndarray):
                    gt_sequence = gt_sequence.tolist()

                # Compute Distance
                dist = compute_levenshtein(pred_sequence, gt_sequence)
                errors.append(dist)

    # Compute Correlation
    if len(errors) > 1:
        correlation, _ = scipy.stats.pearsonr(errors, lengths_list)
        print(
            f"Correlation between Error (Levenshtein Dist) and Sequence Length: {correlation:.4f}"
        )
    else:
        print("Not enough samples for correlation analysis.")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # We use debug=False to train on the full dataset to meet the metric requirement.
    # The dataset is small enough (~1700 train samples) to fit within the time limit.
    print("Initializing Datasets...")
    train_dataset = GestureDataset(split="train", augment=True, debug=False)
    val_dataset = GestureDataset(split="val", augment=False, debug=False)
    test_dataset = GestureDataset(split="test", augment=False, debug=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=GestureDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=GestureDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=GestureDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Training
    trainer = Trainer(device=device)

    # Limit epochs to ensure fast baseline execution while allowing convergence
    print("Starting Training...")
    trainer.fit(train_loader, val_loader, num_epochs=Config.NUM_EPOCHS, patience=5)

    # 4. Validation Metric
    print("Computing Final Validation Metric...")
    _, val_score = trainer.validate(val_loader)
    print(f"Final Validation Metric: {val_score}")

    # 5. Failure Analysis
    analyze_failures(trainer, val_loader, device)

    # 6. Submission
    # Threshold defined in task description logic
    THRESHOLD = 0.08548168249660787

    if val_score < THRESHOLD:
        print(
            f"\nValidation score ({val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        predictions = trainer.predict(test_loader)
        generate_submission(predictions, output_filename="submission.csv")
    else:
        print(
            f"\nValidation score ({val_score}) does not meet threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
