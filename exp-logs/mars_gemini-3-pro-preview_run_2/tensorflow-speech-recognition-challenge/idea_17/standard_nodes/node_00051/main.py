import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import SEED, EPOCHS
from library.utils import set_seed
from library.data_manager import load_dataset_to_memory
from library.engine import Trainer


def main():
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    # 2. Load Data
    # We load the full dataset. The GPU-resident pipeline is fast enough
    # to handle the full 46k samples in minutes on an A100.
    print("Loading dataset...")
    data_dict = load_dataset_to_memory(load_cached_data=True)

    # 3. Initialize Trainer
    trainer = Trainer(data_dict, device=device)

    # 4. Train
    # The fit method handles the training loop, validation, and EMA updates.
    # It saves the best model to checkpoints.
    trainer.fit(epochs=EPOCHS)

    # 5. Final Evaluation & Failure Analysis
    print("\nRunning Final Evaluation and Failure Analysis...")

    # Load best EMA model for evaluation
    best_model_path = trainer.best_model_path
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        trainer.model.load_state_dict(state_dict)
        # Update EMA wrapper to match loaded weights
        trainer.ema.ema.load_state_dict(state_dict)

    ema_model = trainer.ema.get_model()
    ema_model.eval()

    # Get Validation Data from trainer (already on GPU)
    val_wavs = trainer.val_wavs
    val_lbls = trainer.val_lbls

    # Run Inference on Validation Set
    all_preds = []
    all_probs = []
    batch_size = 128  # Larger batch size for inference

    with torch.no_grad():
        num_samples = val_wavs.size(0)
        indices = torch.arange(num_samples, device=device)
        batches = torch.split(indices, batch_size)

        for batch_idx in batches:
            x = val_wavs[batch_idx]
            logits = ema_model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    val_preds = torch.cat(all_preds).to(device)
    val_probs = torch.cat(all_probs).to(device)

    # Calculate Metric
    correct = (val_preds == val_lbls).sum().item()
    total = val_lbls.size(0)
    final_acc = correct / total

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis
    # Calculate error vector (1 for error, 0 for correct)
    errors = (val_preds != val_lbls).float().cpu().numpy()

    # Calculate signal features from raw waveforms (on CPU for scipy)
    # 1. Max Amplitude
    val_wavs_cpu = val_wavs.cpu()
    max_amp = val_wavs_cpu.abs().max(dim=1).values.numpy()

    # 2. RMS Energy
    rms_energy = torch.sqrt(torch.mean(val_wavs_cpu**2, dim=1)).numpy()

    # Calculate correlations
    # Handle potential constant arrays (though unlikely with audio data)
    if np.std(errors) > 0:
        corr_amp, _ = pearsonr(errors, max_amp)
        corr_rms, _ = pearsonr(errors, rms_energy)
    else:
        corr_amp, corr_rms = 0.0, 0.0

    print("\n=== Failure Analysis ===")
    print(f"Correlation between Error and Max Amplitude: {corr_amp:.4f}")
    print(f"Correlation between Error and RMS Energy: {corr_rms:.4f}")

    # Identify top confused classes (optional but helpful log)
    if final_acc < 1.0:
        print("Analyzing confusion patterns...")
        # Simple confusion check
        error_indices = np.where(errors == 1)[0]
        if len(error_indices) > 0:
            print(f"Total Errors: {len(error_indices)}")

    # 6. Submission
    # Threshold check
    THRESHOLD = 0.9867045739610335

    if final_acc > THRESHOLD:
        print(
            f"\nValidation accuracy ({final_acc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict_submission()
    else:
        print(
            f"\nValidation accuracy ({final_acc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
