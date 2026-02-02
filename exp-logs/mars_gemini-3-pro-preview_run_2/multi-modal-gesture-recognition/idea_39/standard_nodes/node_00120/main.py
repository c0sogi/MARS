import os
import torch
import numpy as np
import scipy.stats
from itertools import groupby

from library.config import PATHS, get_hyperparams
from library.utils import set_seed, compute_levenshtein
from library.trainer import Trainer
from library.inference import run_inference


def main():
    # 1. Setup and Initialization
    # Ensure reproducibility
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 2. Training
    print("Initializing Trainer...")
    trainer = Trainer()

    # Adjust hyperparameters for a fast baseline execution
    # Reducing epochs to 30 ensures completion within the time limit while allowing convergence
    trainer.hp["epochs"] = 30

    print("Starting Training...")
    trainer.fit()

    # Retrieve and print the final validation metric
    # The metric is Levenshtein Distance / Total Gestures (Error Rate)
    final_metric = trainer.best_val_score
    print(f"Final Validation Metric: {final_metric}")

    # 3. Failure Analysis
    print("\nStarting Failure Analysis on Validation Set...")

    # Load the best model weights for analysis
    model = trainer.model
    checkpoint_path = PATHS["model_save_path"]
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint.get('epoch', '?')}")
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()
    val_loader = trainer.val_loader

    errors = []
    input_lengths = []

    with torch.no_grad():
        for features, targets, mask, _ in val_loader:
            features = features.to(device)
            mask = mask.to(device)

            # Forward Pass
            outputs = model(features, mask)

            # Use Stage 3 output (index 2) for final predictions
            stage3_out = outputs[2]
            cls_probs = stage3_out[
                :, :21, :
            ]  # First 21 channels are class probabilities

            # Decode predictions
            batch_preds = trainer._decode_predictions(cls_probs, mask)

            # Process Targets
            targets_np = targets.numpy()
            mask_np = mask.cpu().numpy()

            for i in range(len(batch_preds)):
                # Determine valid length of the sequence
                valid_len = int(mask_np[i].sum())

                # Extract Ground Truth Sequence
                # Collapse frame-wise targets to gesture sequence (removing background 0)
                raw_t = targets_np[i, :valid_len]
                t_seq = [k for k, g in groupby(raw_t) if k != 0]

                # Prediction Sequence
                p_seq = batch_preds[i]

                # Compute Levenshtein Distance (Error Magnitude)
                dist = compute_levenshtein(p_seq, t_seq)

                errors.append(dist)
                input_lengths.append(valid_len)

    # Calculate Correlation
    if len(errors) > 1:
        corr, p_val = scipy.stats.pearsonr(errors, input_lengths)
        print(f"Correlation between Error and Sequence Length: {corr:.16f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 4. Submission Generation
    threshold = 0.0586756077116513
    if final_metric < threshold:
        print(
            f"\nValidation metric {final_metric} is lower than threshold {threshold}."
        )
        print("Generating submission file...")
        run_inference(load_cached_data=True)
    else:
        print(
            f"\nValidation metric {final_metric} is not lower than threshold {threshold}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
