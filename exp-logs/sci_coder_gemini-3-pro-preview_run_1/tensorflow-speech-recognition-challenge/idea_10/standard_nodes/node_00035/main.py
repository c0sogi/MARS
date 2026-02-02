import os
import shutil
import pandas as pd
import numpy as np
import torch
import soundfile as sf
import warnings

# Import from provided library files
from library.config import Config
from library.trainer import Trainer
from library.utils import set_seed, LabelMapper

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Optimize Config for A100 GPU and Time Limit
    # Increasing Batch Size to utilize 40GB VRAM and speed up training
    Config.BATCH_SIZE = 128
    # Set Epochs to a reasonable number for convergence within 2 hours
    # The provided Trainer uses a scheduler that adapts to Config.EPOCHS
    Config.EPOCHS = 15
    Config.NUM_WORKERS = 4

    print(
        f"Configuration Configured: Batch Size={Config.BATCH_SIZE}, Epochs={Config.EPOCHS}"
    )

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    # Initialize Trainer (loads cached balanced data if available)
    trainer = Trainer(load_cached_data=True)

    # Start Training
    trainer.fit()

    # -------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("\nRunning Final Validation...")

    # Load the best model checkpoint
    if os.path.exists(Config.CHECKPOINT_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.CHECKPOINT_PATH, map_location=trainer.device)
        )

    # Compute metrics
    val_loss, val_acc = trainer.validate()

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {val_acc}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    trainer.model.eval()
    device = trainer.device
    val_loader = trainer.val_loader

    # Access the underlying data records to get filepaths
    # val_loader.dataset.data is a list of dicts.
    # Since val_loader has shuffle=False, the order matches the iteration.
    dataset_records = val_loader.dataset.data

    all_preds_idx = []
    all_targets_idx = []

    # Collect raw predictions
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = trainer.model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds_idx.extend(preds.cpu().numpy())
            all_targets_idx.extend(targets.cpu().numpy())

    # Analyze Errors
    mapper = LabelMapper()
    errors = []
    durations = []
    label_indices = []

    # Iterate through records to extract features and compute error
    # We limit this to the number of processed samples (in case of drop_last, though val usually doesn't)
    num_samples = len(all_preds_idx)

    for i in range(num_samples):
        record = dataset_records[i]

        # 1. Determine Error (based on 12-class mapping)
        pred_label = mapper.index_to_submission(all_preds_idx[i])
        true_label = mapper.index_to_submission(all_targets_idx[i])

        # Error is 1 if incorrect, 0 if correct
        is_error = 1 if pred_label != true_label else 0
        errors.append(is_error)

        # 2. Feature: Label Index (Proxy for class difficulty)
        label_indices.append(all_targets_idx[i])

        # 3. Feature: Audio Duration
        # Handle virtual silence which doesn't have a physical file in the same way
        if record["filepath"] == "virtual_silence":
            durations.append(1.0)  # Standard duration
        else:
            full_path = os.path.join(Config.INPUT_ROOT, record["filepath"])
            try:
                # Fast header read
                info = sf.info(full_path)
                durations.append(info.duration)
            except Exception:
                durations.append(1.0)  # Fallback

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {"error": errors, "duration": durations, "label_idx": label_indices}
    )

    # Compute Correlations
    corr_duration = df_analysis["error"].corr(df_analysis["duration"])
    corr_label = df_analysis["error"].corr(df_analysis["label_idx"])

    print(f"Correlation between Error and Duration: {corr_duration}")
    print(f"Correlation between Error and Label Index: {corr_label}")

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9872909698996656

    if val_acc > THRESHOLD:
        print(
            f"\nValidation metric ({val_acc}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions (saves to Config.SUBMISSION_PATH)
        trainer.predict()

        # Move to required location ./submission/submission.csv
        output_dir = "./submission"
        os.makedirs(output_dir, exist_ok=True)

        src_path = Config.SUBMISSION_PATH
        dst_path = os.path.join(output_dir, "submission.csv")

        shutil.move(src_path, dst_path)
        print(f"Final submission moved to {dst_path}")

    else:
        print(
            f"\nValidation metric ({val_acc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
