import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.dataset import get_balanced_dataloaders, get_test_dataloader
from library.model import EfficientNetAudio
from library.trainer import Trainer, generate_submission


def calculate_correlation(x, y):
    """Calculates Pearson correlation coefficient between two numpy arrays."""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    # Handle cases with zero variance to avoid division by zero/NaNs
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]


def main():
    # 1. Configuration and Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_balanced_dataloaders()

    # 3. Model Initialization
    print("Initializing EfficientNetAudio model...")
    model = EfficientNetAudio(num_classes=Config.NUM_CLASSES, pretrained=True)

    # 4. Training
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, device=device)
    # Using the Config's default epoch count (20)
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # 5. Evaluation on Validation Set
    print("Loading best model for evaluation...")
    # Load the best checkpoint saved during training
    try:
        load_checkpoint(Config.MODEL_SAVE_PATH, model, device=device)
    except FileNotFoundError:
        print("Warning: Checkpoint not found, using current model state.")

    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate Multiclass Accuracy
    accuracy = np.mean(all_preds == all_targets)
    print(f"Final Validation Metric: {accuracy}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Calculate error magnitude: 1.0 - probability assigned to the correct class
    row_indices = np.arange(len(all_targets))
    correct_class_probs = all_probs[row_indices, all_targets]
    error_magnitudes = 1.0 - correct_class_probs

    # Extract features (Duration) from validation metadata
    val_df = val_loader.dataset.df
    durations = []

    # Iterate over the dataframe to get file paths
    # Note: The order in val_loader (shuffle=False) matches the dataframe order
    for idx, row in val_df.iterrows():
        filepath = row["filepath"]
        full_path = os.path.join(Config.INPUT_ROOT, filepath)

        d = 0.0
        try:
            # sf.info is lightweight, reads header only
            if os.path.exists(full_path):
                info = sf.info(full_path)
                d = info.duration
        except Exception:
            pass
        durations.append(d)

    durations = np.array(durations)

    # Calculate correlation
    corr_duration = calculate_correlation(durations, error_magnitudes)
    print(f"Correlation between Error Magnitude and Audio Duration: {corr_duration}")

    # 7. Submission Generation
    # Threshold from instructions
    THRESHOLD = 0.9611927398444252

    if accuracy > THRESHOLD:
        print(
            f"Validation accuracy ({accuracy}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        test_loader = get_test_dataloader()
        # trainer.predict automatically loads the best checkpoint
        generate_submission(trainer, test_loader)
    else:
        print(
            f"Validation accuracy ({accuracy}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
