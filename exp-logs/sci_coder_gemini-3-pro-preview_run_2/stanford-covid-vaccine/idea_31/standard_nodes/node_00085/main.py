import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.dataset import RNADataset
from library.engine import RNAEngine
from library.model import SR_DCN


def perform_failure_analysis(engine, val_loader, val_metadata_path):
    """
    Analyzes model errors on the validation set to identify systematic weaknesses.
    """
    print("\n==== Failure Analysis ====")

    # 1. Load Best Model
    model = SR_DCN().to(engine.device)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Best model not found, skipping failure analysis.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=engine.device))
    model.eval()

    # 2. Run Inference on Validation Set
    all_ids = []
    all_errors = []

    # Scored columns indices
    scored_indices = engine.scored_indices

    criterion = torch.nn.MSELoss(reduction="none")

    with torch.no_grad():
        for inputs, partner_indices, targets, ids in val_loader:
            inputs = inputs.to(engine.device)
            partner_indices = partner_indices.to(engine.device)
            targets = targets.to(engine.device)

            # Pass 1 -> Pass 2 (Refined)
            preds_1 = model(inputs, partner_indices, recycling=None)
            preds_2 = model(inputs, partner_indices, recycling=preds_1)

            # Slice to scored length and columns
            preds_sliced = preds_2[:, : Config.SCORED_LEN, :][:, :, scored_indices]
            targets_sliced = targets[:, : Config.SCORED_LEN, :][:, :, scored_indices]

            # Calculate MSE per sample (average over sequence and columns)
            # Shape: (Batch, Seq, Cols)
            mse_loss = criterion(preds_sliced, targets_sliced)
            # Mean over seq and cols -> (Batch,)
            sample_mse = mse_loss.mean(dim=(1, 2))
            sample_rmse = torch.sqrt(sample_mse)

            all_ids.extend(ids)
            all_errors.extend(sample_rmse.cpu().numpy())

    # 3. Load Metadata
    df_val = pd.read_csv(val_metadata_path)

    # Create a DataFrame for errors
    df_errors = pd.DataFrame({"id": all_ids, "rmse": all_errors})

    # Merge with metadata
    df_analysis = pd.merge(df_errors, df_val, on="id", how="inner")

    # 4. Feature Engineering for Analysis
    # Base counts
    df_analysis["count_A"] = df_analysis["sequence"].apply(lambda x: x.count("A"))
    df_analysis["count_G"] = df_analysis["sequence"].apply(lambda x: x.count("G"))
    df_analysis["count_C"] = df_analysis["sequence"].apply(lambda x: x.count("C"))
    df_analysis["count_U"] = df_analysis["sequence"].apply(lambda x: x.count("U"))

    # Features to check correlation with
    features_to_check = ["signal_to_noise", "count_A", "count_G", "count_C", "count_U"]

    print(f"{'Feature':<20} | {'Correlation with Error':<20}")
    print("-" * 45)

    for feat in features_to_check:
        if feat in df_analysis.columns:
            # Drop NaNs if any
            valid_data = df_analysis[[feat, "rmse"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["rmse"])
                print(f"{feat:<20} | {corr:.4f}")
            else:
                print(f"{feat:<20} | N/A (Insufficient Data)")
        else:
            print(f"{feat:<20} | Not Found")
    print("-" * 45)


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for Fast Baseline
    Config.EPOCHS = 15  # Reduce epochs for speed
    Config.BATCH_SIZE = 16

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Initializing Datasets...")

    # Train Loader
    train_dataset = RNADataset(mode="train", load_cached_data=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Val Loader
    val_dataset = RNADataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test Loader
    test_dataset = RNADataset(mode="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # =========================================================================
    # 3. Training
    # =========================================================================
    engine = RNAEngine(device=device)

    best_val_score = engine.run_training(train_loader, val_loader, epochs=Config.EPOCHS)

    # =========================================================================
    # 4. Validation Output
    # =========================================================================
    # REQUIRED: Print full precision validation metric
    print(f"Final Validation Metric: {best_val_score}")

    # =========================================================================
    # 5. Failure Analysis
    # =========================================================================
    val_metadata_path = os.path.join(Config.METADATA_DIR, "val.csv")
    perform_failure_analysis(engine, val_loader, val_metadata_path)

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    THRESHOLD = 0.5417620723771521

    if best_val_score < THRESHOLD:
        print(
            f"\nValidation score ({best_val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        engine.generate_submission(test_loader)
    else:
        print(
            f"\nValidation score ({best_val_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
