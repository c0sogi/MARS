import os
import torch
import numpy as np
import random
import warnings
from scipy.stats import pearsonr

# Import from provided library files
from library.config import TRAIN_CONFIG, LABEL_CONFIG
from library.dataset import get_dataloaders
from library.trainer import Trainer

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Configuration and Setup
    # Override config for a fast baseline execution
    TRAIN_CONFIG.epochs = 15
    TRAIN_CONFIG.swa_start_epoch = 10

    set_seed(TRAIN_CONFIG.seed)
    device = torch.device(TRAIN_CONFIG.device)

    print(
        f"Running fast baseline with {TRAIN_CONFIG.epochs} epochs (SWA start: {TRAIN_CONFIG.swa_start_epoch})..."
    )

    # 2. Data Loading
    # Load cached data for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Training
    # Initialize Trainer with test_loader=None to control submission manually
    trainer = Trainer(train_loader, val_loader, test_loader=None)
    trainer.fit()

    # 4. Validation & Failure Analysis
    print("\nStarting Evaluation and Failure Analysis...")

    # Use the final SWA model for evaluation
    model = trainer.swa_model
    model.eval()

    # Metrics storage
    correct_12class = 0
    total_samples = 0

    # Failure analysis storage
    error_magnitudes = []
    rms_values = []

    # Preprocessing transform (LogMelSpectrogram) is part of the trainer
    # We need to apply it manually or use the trainer's reference
    log_mel = trainer.log_mel

    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            # --- Feature Extraction for Failure Analysis ---
            # Calculate RMS (Root Mean Square) of the raw waveform
            # waveform shape: (Batch, Time)
            rms = torch.sqrt(torch.mean(waveforms**2, dim=1))
            rms_values.extend(rms.cpu().numpy())

            # --- Inference ---
            specs = log_mel(waveforms)
            logits = model(specs)
            probs = torch.softmax(logits, dim=1)

            # --- Error Magnitude Calculation ---
            # Probability assigned to the true fine-grained class
            true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            # Error = 1 - Confidence in Truth
            errors = 1.0 - true_probs
            error_magnitudes.extend(errors.cpu().numpy())

            # --- 12-Class Metric Calculation ---
            # Get predicted fine-grained ID
            _, pred_ids = torch.max(probs, dim=1)

            # Convert to CPU for string mapping
            pred_ids_np = pred_ids.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for pred_id, true_id in zip(pred_ids_np, labels_np):
                # Map ID -> Fine Label -> Submission Label
                pred_label_fine = LABEL_CONFIG.id2label[pred_id]
                true_label_fine = LABEL_CONFIG.id2label[true_id]

                pred_label_sub = LABEL_CONFIG.map_to_submission_label(pred_label_fine)
                true_label_sub = LABEL_CONFIG.map_to_submission_label(true_label_fine)

                if pred_label_sub == true_label_sub:
                    correct_12class += 1
                total_samples += 1

    # Calculate Final Metric
    final_metric = correct_12class / total_samples
    print(f"Final Validation Metric: {final_metric}")

    # Calculate Correlations
    if len(error_magnitudes) > 1:
        corr_rms, _ = pearsonr(error_magnitudes, rms_values)
        print(f"Correlation (Error Magnitude vs RMS): {corr_rms:.4f}")
    else:
        print("Not enough samples for correlation analysis.")

    # 5. Submission
    THRESHOLD = 0.9872909698996656

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        # Assign test loader and generate
        trainer.test_loader = test_loader
        trainer.generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
