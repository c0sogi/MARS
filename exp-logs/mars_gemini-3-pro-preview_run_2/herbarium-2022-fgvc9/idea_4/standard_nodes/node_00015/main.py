import os
import sys
import pandas as pd
import numpy as np
import torch
from library.config import Config, seed_everything
from library.utils import get_logger
from library.dataset import get_dataloaders
from library.trainer import Trainer
from library.inference import InferenceRunner

# ==========================================
# 1. Configuration & Setup
# ==========================================
# Override Config for Fast Baseline Execution
Config.EPOCHS = 1
Config.BATCH_SIZE = 96  # Adjusted for A100 memory safety with B4
Config.DEBUG = False  # We will manually subset data instead of using library debug mode

# Setup Logger
logger = get_logger("baseline")
seed_everything(Config.SEED)


def create_training_subset(fraction=0.5):
    """Creates a temporary subset of the training data to speed up the epoch."""
    logger.info(f"Creating {fraction*100}% training subset for speed...")
    original_train_path = os.path.join(Config.METADATA_DIR, "train.csv")
    subset_path = os.path.join(Config.WORK_DIR, "train_subset_runtime.csv")

    df = pd.read_csv(original_train_path)
    # Stratified sample if possible, else random
    try:
        subset_df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(frac=fraction, random_state=Config.SEED)
        )
    except:
        subset_df = df.sample(frac=fraction, random_state=Config.SEED)

    subset_df.to_csv(subset_path, index=False)
    logger.info(f"Subset saved to {subset_path} with {len(subset_df)} samples.")
    return subset_path


def main():
    # 2. Prepare Data
    # Point Config to our smaller subset
    Config.TRAIN_CSV = create_training_subset(fraction=0.5)

    logger.info("Loading DataLoaders...")
    train_loader, val_loader, test_loader, meta_counts = get_dataloaders(
        debug=Config.DEBUG
    )

    # 3. Training
    logger.info("Initializing Trainer...")
    trainer = Trainer(meta_counts)

    logger.info("Starting Training Loop...")
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 4. Final Validation Assessment
    logger.info("Performing Final Validation...")
    val_loss, val_f1 = trainer.validate(val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_f1}")

    # 5. Failure Analysis
    logger.info("Running Failure Analysis...")
    model = trainer.model
    model.eval()
    device = Config.DEVICE

    all_preds = []
    all_labels = []

    # Custom inference loop to get raw predictions for analysis
    with torch.no_grad():
        for images, (species_labels, _, _) in val_loader:
            images = images.to(device)
            # labels=None -> ArcFace returns scaled cosine logits
            outputs = model(images, labels=None)
            sp_logits = outputs[0]
            _, preds = torch.max(sp_logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(species_labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Binary Error: 1 if Incorrect, 0 if Correct
    # (Note: Usually failure analysis correlates Error (1) with features)
    is_incorrect = (all_preds != all_labels).astype(int)

    # Feature: Class Frequency (from training distribution)
    train_df = pd.read_csv(Config.TRAIN_CSV)
    class_counts_map = train_df["label"].value_counts().to_dict()

    # Map counts to validation samples
    val_class_counts = np.array([class_counts_map.get(l, 0) for l in all_labels])

    # Compute Correlation
    if len(np.unique(is_incorrect)) > 1:
        correlation = np.corrcoef(is_incorrect, val_class_counts)[0, 1]
        print(
            f"Correlation between Error (1=Wrong) and Class Frequency: {correlation:.6f}"
        )
    else:
        print("Correlation undefined (variance is zero).")

    # 6. Submission Generation
    # Strict threshold check as per requirements
    THRESHOLD = 0.6021914648406147

    if val_f1 > THRESHOLD:
        logger.info(
            f"Validation F1 ({val_f1:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        runner = InferenceRunner()
        runner.run()
    else:
        logger.info(
            f"Validation F1 ({val_f1:.6f}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
