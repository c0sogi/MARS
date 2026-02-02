import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler

# Import provided library modules
from library.config import Config
from library.data import get_dataloaders, process_metadata
from library.model import CMTSINModel
from library.losses import MultiTaskLoss
from library.train import train_one_epoch, validate, set_seed
from library.inference import predict_and_submit


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Adjust configuration for a fast baseline execution
    Config.NUM_EPOCHS = 2  # Limit epochs to ensure completion within strict time limits
    Config.DEBUG = False  # Use full dataset to ensure we beat the threshold

    # Dynamically scale hyperparameters for 16GB GPU (Cite debug_lesson_3)
    # Override config to ensure changes apply even if module is cached (Cite debug_lesson_2)
    Config.IMAGE_SIZE = (512, 512)
    Config.BATCH_SIZE = 12
    Config.NUM_WORKERS = 4

    # Set random seeds for reproducibility
    set_seed(Config.SEED)

    # Setup device
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Loading DataLoaders...")
    # load_cached_data=True allows using preprocessed parquets if available
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print(f"Initializing Model: {Config.MODEL_NAME}")
    model = CMTSINModel()
    model.to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    # Loss Function and Scaler for AMP
    loss_fn = MultiTaskLoss()
    loss_fn.to(device)
    scaler = GradScaler()

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    print(f"Starting Training for {Config.NUM_EPOCHS} epochs...")
    best_pf1 = -1.0

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, epoch
        )

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, loss_fn, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val pF1: {val_pf1:.6f}"
        )

        # Save Best Model
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  [+] Saved new best model with pF1: {best_pf1:.6f}")

    # =========================================================================
    # 5. Final Evaluation & Failure Analysis
    # =========================================================================
    print(f"Final Validation Metric: {best_pf1}")

    print("\nRunning Failure Analysis on Validation Set...")
    # Load best weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Collect predictions and metadata
    all_preds = []
    all_targets = []
    all_image_ids = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)
            targets = batch["targets"]["cancer"].to(device)
            image_ids = batch["image_id"]  # Tensor of ints

            # Inference
            outputs = model(images, meta)
            probs = torch.sigmoid(outputs["cancer"]).cpu().numpy()
            targs = targets.cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targs)
            all_image_ids.extend(image_ids.numpy())

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(
        {"image_id": all_image_ids, "prob": all_preds, "target": all_targets}
    )
    analysis_df["error"] = np.abs(analysis_df["prob"] - analysis_df["target"])

    # Load metadata to merge features (Age, Density, etc.)
    # process_metadata returns (train, val, test) dataframes
    _, val_meta_df, _ = process_metadata(load_cached_data=True)

    # Merge on image_id
    full_analysis = pd.merge(analysis_df, val_meta_df, on="image_id", how="left")

    # Calculate Correlations
    print("Correlations between Error Magnitude and Features:")

    # 1. Age
    if "age" in full_analysis.columns:
        corr = full_analysis["error"].corr(full_analysis["age"])
        print(f"  Error vs Age: {corr:.4f}")

    # 2. Density (using target_density which maps A-D to 0-3)
    if "target_density" in full_analysis.columns:
        # Filter out missing values (-100)
        mask = full_analysis["target_density"] != -100
        if mask.sum() > 0:
            corr = full_analysis.loc[mask, "error"].corr(
                full_analysis.loc[mask, "target_density"]
            )
            print(f"  Error vs Density: {corr:.4f}")

    # 3. Machine ID
    if "machine_idx" in full_analysis.columns:
        corr = full_analysis["error"].corr(full_analysis["machine_idx"])
        print(f"  Error vs Machine ID: {corr:.4f}")

    # =========================================================================
    # 6. Submission
    # =========================================================================
    THRESHOLD = 0.044888656586408615

    if best_pf1 > THRESHOLD:
        print(
            f"\nValidation metric ({best_pf1}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit()
    else:
        print(
            f"\nValidation metric ({best_pf1}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
