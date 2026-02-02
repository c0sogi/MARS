import os
import sys
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from scipy.stats import pointbiserialr
from sklearn.metrics import accuracy_score

# Import library modules
from library.config import Config
from library.utils import set_seed, init_logger
from library.trainer import Trainer
from library.dataset import get_dataloaders
from library.model import MultiResConvNeXtCRNN


def main():
    # 1. Setup and Configuration overrides for fast baseline
    # Limit epochs to ensure completion within 2 hours while allowing convergence
    Config.EPOCHS = 12
    Config.setup()
    set_seed(Config.SEED)

    # Initialize Logger
    logger = init_logger()
    logger.info("Starting runfile.py execution...")

    # 2. Training
    trainer = Trainer()
    trainer.fit(epochs=Config.EPOCHS)

    # 3. Detailed Validation & Failure Analysis
    logger.info("Performing detailed validation and failure analysis...")

    # Load best model
    device = Config.DEVICE
    model = MultiResConvNeXtCRNN().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        logger.info(f"Loaded best model from {Config.MODEL_SAVE_PATH}")
    else:
        logger.warning("Best model not found, using current model state.")

    model.eval()

    # Get validation loader
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    all_preds = []
    all_labels = []
    all_filepaths = []

    # Inference loop (No Grad)
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Get filepaths for this batch
            # The dataset inside loader is accessible
            start_idx = i * val_loader.batch_size
            end_idx = start_idx + labels.size(0)
            # Depending on drop_last or shuffle, strict indexing might vary,
            # but val_loader is shuffle=False, drop_last=False.
            # We can access the dataset directly via indices if needed,
            # but cleaner to rely on batch iteration if dataset supported it.
            # Since dataset.get_filename exists:
            batch_filepaths = [
                val_loader.dataset.get_filename(idx)
                for idx in range(start_idx, end_idx)
            ]
            all_filepaths.extend(batch_filepaths)

    # Compute Metric
    val_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Final Validation Metric: {val_accuracy}")

    # Failure Analysis
    # Calculate correlation between Error (1=Wrong, 0=Right) and Duration
    errors = (np.array(all_preds) != np.array(all_labels)).astype(int)
    durations = []

    logger.info("Extracting features for failure analysis...")
    for fp in all_filepaths:
        full_path = os.path.join(Config.INPUT_DIR, fp)
        try:
            info = sf.info(full_path)
            durations.append(info.duration)
        except:
            durations.append(0.0)

    if len(set(errors)) > 1:
        # Point Biserial Correlation: Binary variable (Error) vs Continuous variable (Duration)
        corr, p_value = pointbiserialr(errors, durations)
        print(
            f"Correlation between Error and Audio Duration: {corr:.4f} (p-value: {p_value:.4f})"
        )
    else:
        print(
            "Correlation between Error and Audio Duration: Undefined (All predictions correct or all wrong)"
        )

    # 4. Submission
    threshold = 0.9769230769230769
    if val_accuracy > threshold:
        logger.info(
            f"Validation accuracy {val_accuracy} > {threshold}. Generating submission..."
        )
        trainer.predict()
    else:
        logger.info(
            f"Validation accuracy {val_accuracy} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
