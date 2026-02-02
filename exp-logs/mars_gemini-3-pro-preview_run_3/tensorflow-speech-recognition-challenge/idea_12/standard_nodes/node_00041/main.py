import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.preprocessor import Preprocessor
from library.trainer import Trainer
from library.utils import load_checkpoint, set_seed
from library.dataset import CachedSpeechDataset
from library.model import FrequencyPreservingSKResNetCRNN


def main():
    # 1. Setup and Config Overrides for Fast Baseline
    # Limit epochs to ensure execution finishes well within the time limit
    Config.EPOCHS = 12
    Config.PATIENCE = 4

    # Ensure fully reproducible results
    set_seed(Config.SEED)

    # 2. Preprocessing
    # Ensure features are cached. load_cached_data=True uses existing files if available.
    preprocessor = Preprocessor()
    preprocessor.cache_dataset(load_cached_data=True)

    # 3. Training
    # Initialize trainer and start training loop
    trainer = Trainer()
    trainer.fit()

    # 4. Load Best Model for Evaluation
    # Re-initialize model structure
    model = FrequencyPreservingSKResNetCRNN().to(Config.DEVICE)

    # Load weights from the best checkpoint saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        epoch, score = load_checkpoint(
            model, path=Config.MODEL_SAVE_PATH, device=Config.DEVICE
        )
        print(f"Loaded best model from epoch {epoch} with val score {score}")
    else:
        print("No checkpoint found. Using current model weights.")

    model.eval()

    # 5. Validation & Failure Analysis
    val_dataset = CachedSpeechDataset(Config.VAL_META, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []
    all_targets = []
    input_energies = []

    print("Running validation inference...")
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            # Forward pass
            outputs = model(features)
            _, preds = torch.max(outputs, dim=1)

            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Failure Analysis Feature: Mean Signal Energy
            # features shape: (Batch, Channels, Freq, Time)
            # Calculate mean intensity per sample
            batch_energy = features.view(features.size(0), -1).mean(dim=1).cpu().numpy()
            input_energies.extend(batch_energy)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    input_energies = np.array(input_energies)

    # Calculate Metric
    accuracy = (all_preds == all_targets).mean()
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis: Correlation
    errors = (all_preds != all_targets).astype(int)

    # Avoid correlation calculation if variance is 0 (all correct or all wrong)
    if len(np.unique(errors)) > 1 and len(np.unique(input_energies)) > 1:
        corr_energy = np.corrcoef(errors, input_energies)[0, 1]
        print(f"Correlation between Error and Input Mean Energy: {corr_energy}")
    else:
        print("Correlation between Error and Input Mean Energy: N/A (Zero Variance)")

    # 6. Submission
    # Threshold defined in task
    THRESHOLD = 0.9832324978392394

    if accuracy > THRESHOLD:
        print(f"Validation accuracy {accuracy} > {THRESHOLD}. Generating submission...")

        # Load Test Data
        test_dataset = CachedSpeechDataset(Config.TEST_META, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds = []

        # Inference on Test Set
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(Config.DEVICE)
                outputs = model(features)
                _, preds = torch.max(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy())

        # Map IDs to Labels
        id2label = Config.ID2LABEL
        pred_labels = [id2label[p] for p in test_preds]

        # Get filenames from dataset dataframe
        # The dataset stores the full relative path, we need just the filename
        test_fnames = (
            test_dataset.df["filepath"].apply(lambda x: os.path.basename(x)).tolist()
        )

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"fname": test_fnames, "label": pred_labels})

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation accuracy {accuracy} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
