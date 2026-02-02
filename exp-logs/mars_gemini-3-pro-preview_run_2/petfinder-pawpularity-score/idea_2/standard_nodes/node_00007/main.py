import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import seed_everything, get_rmse_score
from library.dataset import PawpularityDataset, get_transforms
from library.model import PawpularitySwinModel
from library.trainer import run_training, predict


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    seed_everything(42)

    # Paths
    TRAIN_CSV = "./metadata/train.csv"
    VAL_CSV = "./metadata/validation.csv"
    TEST_CSV = "./metadata/test.csv"
    OUTPUT_DIR = "./working/idea_2"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Hyperparameters
    # Increased epochs to 6 to allow the larger model to converge.
    # The dataset size (~7k) allows for quick epochs on GPU.
    EPOCHS = 6
    BATCH_SIZE = 32
    LR_BACKBONE = 1e-5
    LR_HEAD = 1e-4
    PATIENCE = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on device: {DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print("\nStarting Training Pipeline...")
    best_model_path = run_training(
        train_csv_path=TRAIN_CSV,
        val_csv_path=VAL_CSV,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate_backbone=LR_BACKBONE,
        learning_rate_head=LR_HEAD,
        patience=PATIENCE,
        device=DEVICE,
        debug=False,
    )

    # -------------------------------------------------------------------------
    # 3. Validation Assessment
    # -------------------------------------------------------------------------
    print("\nPerforming Final Validation Assessment...")

    # Load validation data
    val_df = pd.read_csv(VAL_CSV)
    val_dataset = PawpularityDataset(val_df, transforms=get_transforms("valid"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load the best model
    model = PawpularitySwinModel()
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    preds = []
    actuals = []

    # Inference loop
    with torch.no_grad():
        for images, metadata, targets in val_loader:
            images = images.to(DEVICE)
            metadata = metadata.to(DEVICE)

            outputs = model(images, metadata)

            # Rescale predictions and targets to [0, 100]
            # Model outputs [0, 1], Dataset targets are [0, 1]
            batch_preds = outputs.cpu().numpy().flatten() * 100.0
            batch_targets = targets.numpy().flatten() * 100.0

            preds.extend(batch_preds)
            actuals.extend(batch_targets)

    preds = np.array(preds)
    actuals = np.array(actuals)

    # Clip predictions to valid range [1, 100]
    preds = np.clip(preds, 1.0, 100.0)

    # Calculate and print Final Metric
    final_rmse = get_rmse_score(preds, actuals)
    print(f"Final Validation Metric: {final_rmse}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nFailure Analysis (Correlation of Error with Features):")

    # Calculate absolute errors
    errors = np.abs(preds - actuals)

    # Identify feature columns for analysis
    exclude_cols = ["Id", "file_path", "pawpularity_bins"]
    analysis_cols = [c for c in val_df.columns if c not in exclude_cols]

    print(f"{'Feature':<20} | {'Correlation':<15}")
    print("-" * 40)

    for col in analysis_cols:
        feat_values = val_df[col].values

        # Calculate Pearson correlation if feature is not constant
        if len(np.unique(feat_values)) > 1:
            corr, _ = pearsonr(feat_values, errors)
            print(f"{col:<20} | {corr:.4f}")
        else:
            print(f"{col:<20} | N/A (Constant)")

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 18.404018195551412

    if final_rmse < THRESHOLD:
        print(
            f"\nValidation metric ({final_rmse}) is better than threshold ({THRESHOLD})."
        )
        print("Generating submission file...")
        predict(
            model_path=best_model_path,
            test_csv_path=TEST_CSV,
            submission_path=SUBMISSION_PATH,
            batch_size=BATCH_SIZE,
            device=DEVICE,
        )
    else:
        print(
            f"\nValidation metric ({final_rmse}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
