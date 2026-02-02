import os
import sys
import torch
import numpy as np
from library.config import Config
from library.trainer import Trainer
from library.data_utils import set_seed


def main():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Adjust configuration for the run
    # We use 50 epochs to allow Model EMA to fully stabilize and converge.
    Config.EPOCHS = 50
    Config.DEBUG = False  # Ensure we use the full dataset to meet the high accuracy bar

    # 2. Training
    # Initialize Trainer
    trainer = Trainer()

    # Run training loop
    # This handles loading data to GPU, training, and saving the best model
    trainer.fit()

    # 3. Validation and Failure Analysis
    print("\n=== Starting Validation and Failure Analysis ===")

    # Ensure the best model is loaded for analysis
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Error: Best model checkpoint not found.")
        sys.exit(1)

    # Load the best state dict into the existing model
    # We reuse trainer.model which already has the correct buffers initialized
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint)
    trainer.model.eval()

    # Access validation data (already on GPU)
    val_wavs = trainer.val_waveforms
    val_lbls = trainer.val_labels

    # Run Inference on Validation Set
    all_probs = []
    batch_size = Config.BATCH_SIZE

    with torch.no_grad():
        for i in range(0, len(val_wavs), batch_size):
            batch_x = val_wavs[i : i + batch_size]
            logits = trainer.model(batch_x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)

    all_probs = torch.cat(all_probs, dim=0)

    # Calculate Final Validation Metric (Accuracy)
    preds = torch.argmax(all_probs, dim=1)
    correct = (preds == val_lbls).sum().item()
    total = len(val_lbls)
    accuracy = correct / total

    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    # Calculate Error Magnitude: 1.0 - Probability of the true class
    # gather: select the probability corresponding to the true label index
    true_class_probs = all_probs.gather(1, val_lbls.unsqueeze(1)).squeeze()
    error_magnitude = 1.0 - true_class_probs

    # Calculate Input Features for Correlation Analysis
    # Feature 1: Max Amplitude (measure of peak volume)
    max_amp = val_wavs.abs().max(dim=1).values

    # Feature 2: Signal Energy (measure of total loudness/power)
    energy = val_wavs.pow(2).mean(dim=1)

    # Move to CPU for numpy correlation calculation
    err_np = error_magnitude.cpu().numpy()
    amp_np = max_amp.cpu().numpy()
    eng_np = energy.cpu().numpy()

    # Calculate Pearson Correlation
    corr_amp = np.corrcoef(err_np, amp_np)[0, 1]
    corr_eng = np.corrcoef(err_np, eng_np)[0, 1]

    print(f"Correlation between Error Magnitude and Max Amplitude: {corr_amp}")
    print(f"Correlation between Error Magnitude and Signal Energy: {corr_eng}")

    # 4. Conditional Submission
    THRESHOLD = 0.9866209549293419

    if accuracy > THRESHOLD:
        print(f"\nValidation accuracy ({accuracy}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        trainer.predict_and_submit()
    else:
        print(
            f"\nValidation accuracy ({accuracy}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
