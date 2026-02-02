import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device, mcrmse
from library.dataset import get_dataloaders
from library.model import RNAGRUNet
from library.engine import Engine


def analyze_failures(engine, val_loader):
    """
    Performs failure analysis on the validation set by correlating
    prediction errors with sample metadata.
    """
    print("Running Failure Analysis...")

    # 1. Get Predictions and Targets
    engine.model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    device = engine.device

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sequence"].to(device)
            struct = batch["structure"].to(device)
            loop = batch["loop_type"].to(device)
            targets = batch["targets"].cpu().numpy()  # (B, 68, 5)
            ids = batch["id"]

            preds = engine.model(seq, struct, loop)
            preds = preds.cpu().numpy()  # (B, 107, 5)

            # Slice predictions to match scored length (68)
            preds_sliced = preds[:, : Config.SEQ_SCORED, :]

            all_preds.append(preds_sliced)
            all_targets.append(targets)
            all_ids.extend(ids)

    # Concatenate
    y_pred = np.concatenate(all_preds, axis=0)  # (N, 68, 5)
    y_true = np.concatenate(all_targets, axis=0)  # (N, 68, 5)

    # 2. Calculate Per-Sample RMSE
    # We use the scored columns defined in Config for the metric
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Indices in TARGET_COLS: reactivity(0), deg_Mg_pH10(1), deg_pH10(2), deg_Mg_50C(3), deg_50C(4)
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    y_pred_scored = y_pred[:, :, scored_indices]
    y_true_scored = y_true[:, :, scored_indices]

    # Calculate MSE per sample (mean over length and channels)
    # Shape: (N, 68, 3) -> (N,)
    mse_per_sample = np.mean((y_true_scored - y_pred_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create Error DataFrame
    df_errors = pd.DataFrame({"id": all_ids, "rmse": rmse_per_sample})

    # 3. Load Metadata to correlate
    if os.path.exists(Config.VAL_FILE):
        df_val_meta = pd.read_parquet(Config.VAL_FILE)

        # Merge on ID
        df_analysis = pd.merge(df_errors, df_val_meta, on="id", how="inner")

        # Calculate Correlations
        # Features of interest
        features = ["signal_to_noise", "SN_filter"]

        # Add derived features
        df_analysis["len_A"] = df_analysis["sequence"].apply(lambda x: x.count("A"))
        df_analysis["len_G"] = df_analysis["sequence"].apply(lambda x: x.count("G"))
        df_analysis["len_C"] = df_analysis["sequence"].apply(lambda x: x.count("C"))
        df_analysis["len_U"] = df_analysis["sequence"].apply(lambda x: x.count("U"))

        features.extend(["len_A", "len_G", "len_C", "len_U"])

        print(f"\nCorrelation with Error (RMSE):")
        for feat in features:
            if feat in df_analysis.columns:
                # Handle potential NaNs just in case, though data analysis showed none
                valid_data = df_analysis[[feat, "rmse"]].dropna()
                if len(valid_data) > 1:
                    corr, _ = pearsonr(valid_data[feat], valid_data["rmse"])
                    print(f"  {feat}: {corr:.4f}")
    else:
        print("Validation metadata file not found. Skipping correlation analysis.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup_directories()

    # Override Config for task requirements
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Fast baseline settings
    EPOCHS = 40

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # 3. Model Setup
    device = get_device()
    model = RNAGRUNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 4. Training
    engine = Engine(model, optimizer, scheduler, device)
    engine.fit(train_loader, val_loader, epochs=EPOCHS, patience=5)

    # 5. Validation Metric
    # Engine.fit loads the best model state automatically at the end
    val_score = engine.validate(val_loader)
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    analyze_failures(engine, val_loader)

    # 7. Submission
    if val_score < 0.7462618350982666:
        print("\nGenerating Submission...")
        engine.predict(test_loader)
    else:
        print(
            f"\nValidation score {val_score} not better than threshold. Skipping submission."
        )


if __name__ == "__main__":
    main()
