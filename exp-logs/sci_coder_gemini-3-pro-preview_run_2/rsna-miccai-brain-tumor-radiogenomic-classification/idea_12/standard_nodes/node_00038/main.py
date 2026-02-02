import os
import pandas as pd
import numpy as np
import torch

# Import provided library modules
from library.utils import seed_everything
from library.dataset import get_dataloader
from library.trainer import Trainer


def main():
    # 1. Setup
    seed_everything(42)

    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 10  # Fast baseline, small dataset allows for quick epochs
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_PATH = "./submission/submission.csv"
    THRESHOLD = 0.6254545454545455

    print("Initializing Pipeline...")

    # 2. Data Loading
    # Load metadata dataframes
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Create DataLoaders
    # Note: We use the working dir for caching ROI calculations to speed up subsequent runs
    train_loader = get_dataloader(
        train_df,
        phase="train",
        batch_size=BATCH_SIZE,
        input_root=INPUT_ROOT,
        cache_dir=WORKING_DIR,
    )
    val_loader = get_dataloader(
        val_df,
        phase="valid",
        batch_size=BATCH_SIZE,
        input_root=INPUT_ROOT,
        cache_dir=WORKING_DIR,
    )
    test_loader = get_dataloader(
        test_df,
        phase="test",
        batch_size=BATCH_SIZE,
        input_root=INPUT_ROOT,
        cache_dir=WORKING_DIR,
    )

    # 3. Training
    config = {
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "checkpoint_dir": WORKING_DIR,
    }

    trainer = Trainer(config=config)

    print("Starting Training...")
    # fit() returns the best validation AUC achieved
    best_auc = trainer.fit(train_loader, val_loader, epochs=EPOCHS, patience=5)

    # 4. Validation Metric
    # Must print strictly in this format
    print(f"Final Validation Metric: {best_auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Load the best model for analysis
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=trainer.device)
        )
    trainer.model.eval()

    val_preds = []
    val_targets = []

    # Run inference on validation set
    # Note: val_loader is not shuffled, so order matches val_df
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(trainer.device, dtype=torch.float32)
            outputs = trainer.model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            val_preds.extend(probs)
            val_targets.extend(targets.numpy().flatten())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate Error
    errors = np.abs(val_targets - val_preds)

    # Extract Feature: FLAIR Slice Count
    # We iterate through the dataframe to get file counts as a proxy for scan resolution/volume
    slice_counts = []
    for _, row in val_df.iterrows():
        flair_path = os.path.join(INPUT_ROOT, row["path_FLAIR"])
        try:
            # Count files in directory
            if os.path.exists(flair_path):
                count = len([f for f in os.listdir(flair_path) if f.endswith(".dcm")])
            else:
                count = 0
        except Exception:
            count = 0
        slice_counts.append(count)

    slice_counts = np.array(slice_counts)

    # Ensure lengths match (loader might drop last if configured, but valid usually doesn't)
    # get_dataloader sets drop_last=False for valid.
    min_len = min(len(errors), len(slice_counts))
    errors = errors[:min_len]
    slice_counts = slice_counts[:min_len]

    # Calculate Correlation
    if len(errors) > 1 and np.std(slice_counts) > 0:
        corr_matrix = np.corrcoef(errors, slice_counts)
        correlation = corr_matrix[0, 1]
        print(f"Correlation between Error and FLAIR Slice Count: {correlation}")
    else:
        print(
            "Could not calculate correlation (insufficient data or constant variance)."
        )

    # 6. Submission
    if best_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({best_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict(test_loader, output_path=SUBMISSION_PATH)
    else:
        print(
            f"\nValidation AUC ({best_auc}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
