import os
import sys
import torch
import pandas as pd
import numpy as np
import cv2
import warnings
from scipy.stats import spearmanr

# Import from provided libraries
from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import get_dataloaders
from library.model import RetinopathyModel
from library.engine import fit_phase, evaluate, predict


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    warnings.filterwarnings("ignore")
    seed_everything(42)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Phase 1: Coarse Training (512x512)
    # ==========================================
    print("\n=== Starting Phase 1: Coarse Training (512x512) ===")

    # Hyperparameters for Phase 1
    IMG_SIZE_1 = 512
    # Reduced batch size to fit in 16GB VRAM.
    # Effective Batch Size = BATCH_SIZE_1 * ACCUM_STEPS_1 = 8 * 3 = 24
    BATCH_SIZE_1 = 8
    ACCUM_STEPS_1 = 3
    EPOCHS_1 = 4
    LR_1 = 1e-4

    # Load Data
    print("Loading 512x512 Data...")
    train_loader_1, val_loader_1, _ = get_dataloaders(
        image_size=IMG_SIZE_1,
        batch_size=BATCH_SIZE_1,
        num_workers=2,
        load_cached_data=True,
        base_path=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        cache_dir=WORKING_DIR,
    )

    # Initialize Model
    print("Initializing EfficientNet-B5 Model...")
    model = RetinopathyModel(
        model_name="efficientnet_b5", pretrained=True, drop_rate=0.5
    )
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # Train
    save_path_1 = os.path.join(WORKING_DIR, "model_512.pth")
    best_kappa_1 = fit_phase(
        model=model,
        train_loader=train_loader_1,
        val_loader=val_loader_1,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS_1,
        save_path=save_path_1,
        patience=2,
        scheduler=scheduler,
        accumulation_steps=ACCUM_STEPS_1,
    )

    print(f"Phase 1 Best Kappa: {best_kappa_1}")

    # Cleanup to save memory
    del train_loader_1, val_loader_1, optimizer, scheduler
    torch.cuda.empty_cache()

    # ==========================================
    # 3. Phase 2: Fine-Tuning (768x768)
    # ==========================================
    print("\n=== Starting Phase 2: Fine-Tuning (768x768) ===")

    # Hyperparameters for Phase 2
    IMG_SIZE_2 = 768
    # Drastically reduced batch size for high resolution.
    # Effective Batch Size = BATCH_SIZE_2 * ACCUM_STEPS_2 = 2 * 4 = 8
    BATCH_SIZE_2 = 2
    ACCUM_STEPS_2 = 4
    EPOCHS_2 = 2  # Fewer epochs for fine-tuning
    LR_2 = 1e-5  # Reduced learning rate

    # Load Data
    print("Loading 768x768 Data...")
    train_loader_2, val_loader_2, test_loader_2 = get_dataloaders(
        image_size=IMG_SIZE_2,
        batch_size=BATCH_SIZE_2,
        num_workers=2,
        load_cached_data=True,
        base_path=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        cache_dir=WORKING_DIR,
    )

    # Note: Model already contains best weights from Phase 1 (loaded by fit_phase)
    # The Global Average Pooling layer handles the change in input resolution automatically.

    # Re-initialize Optimizer with lower LR
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # Train
    save_path_2 = os.path.join(WORKING_DIR, "model_768.pth")
    best_kappa_2 = fit_phase(
        model=model,
        train_loader=train_loader_2,
        val_loader=val_loader_2,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS_2,
        save_path=save_path_2,
        patience=2,
        scheduler=scheduler,
        accumulation_steps=ACCUM_STEPS_2,
    )

    print(f"Phase 2 Best Kappa: {best_kappa_2}")

    # ==========================================
    # 4. Validation & Failure Analysis
    # ==========================================
    print("\n=== Validation & Failure Analysis ===")

    # Generate predictions on validation set for detailed analysis
    model.eval()
    val_preds = []
    val_targets = []

    # We iterate manually to ensure we capture all data without drop_last issues
    # (though val_loader usually has drop_last=False)
    with torch.no_grad():
        for images, labels in val_loader_2:
            images = images.to(device)
            outputs = model(images)
            val_preds.extend(outputs.cpu().numpy().flatten())
            val_targets.extend(labels.numpy().flatten())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Compute and Print Final Metric
    final_kappa = quadratic_weighted_kappa(val_targets, val_preds)
    print(f"Final Validation Metric: {final_kappa}")

    # Failure Analysis: Correlation of Error with Metadata
    print("Performing Failure Analysis...")

    # Load validation metadata to access file paths
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))

    # Calculate absolute error (Regression L1 error)
    errors = np.abs(val_targets - val_preds)

    # Extract meta-features from disk
    widths = []
    heights = []
    intensities = []

    # Limit analysis to the number of predictions we have
    n_samples = len(errors)

    for idx in range(n_samples):
        row = val_df.iloc[idx]
        fpath = os.path.join(INPUT_DIR, row["file_path"])

        # Default values
        w, h, i = 0, 0, 0

        try:
            # Read image to get original dimensions and intensity
            # We use a quick read; if it fails, use 0
            img = cv2.imread(fpath)
            if img is not None:
                h, w, _ = img.shape
                i = img.mean()
        except Exception:
            pass

        widths.append(w)
        heights.append(h)
        intensities.append(i)

    # Create analysis dataframe
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "width": widths,
            "height": heights,
            "intensity": intensities,
            "target": val_targets,
        }
    )

    print("Correlation between Model Error and Input Features (Spearman):")
    for feat in ["width", "height", "intensity", "target"]:
        corr, _ = spearmanr(df_analysis["error"], df_analysis[feat])
        print(f"  {feat}: {corr:.4f}")

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD = 0.9147885422881397

    if final_kappa > THRESHOLD:
        print(
            f"\nValidation metric ({final_kappa}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_preds_raw = predict(model, test_loader_2, device)

        # Post-process: Round to nearest integer and clip to [0, 4]
        test_preds = np.clip(np.round(test_preds_raw), 0, 4).astype(int)

        # Load test metadata for ID mapping
        test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

        # Ensure lengths match
        if len(test_preds) != len(test_df):
            print(
                f"Warning: Prediction count ({len(test_preds)}) matches test set size ({len(test_df)})?"
            )
            # Truncate or pad if necessary, though logic guarantees match
            min_len = min(len(test_preds), len(test_df))
            test_preds = test_preds[:min_len]
            test_df = test_df.iloc[:min_len]

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"id_code": test_df["id_code"], "diagnosis": test_preds}
        )

        # Save
        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nValidation metric ({final_kappa}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
