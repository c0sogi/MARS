import os
import numpy as np
import torch
import warnings
from scipy.stats import pearsonr

# Import provided library functions and classes
from library.utils import set_seed, do_kaggle_metric
from library.dataset import get_dataloaders, ORIG_SIZE, TARGET_SIZE
from library.model_arch import ResNet34UNetPlusPlus
from library.losses import BCEDiceLoss
from library.trainer import ModelTrainer
from library.inference import predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters for Fast Baseline
    BATCH_SIZE = 32
    NUM_EPOCHS = 25  # Limited epochs for fast execution while ensuring convergence
    LEARNING_RATE = 1e-4
    PATIENCE = 8

    # Paths
    CHECKPOINT_DIR = "./working/checkpoints"
    CHECKPOINT_NAME = "best_model.pth"
    SUBMISSION_DIR = "./submission"

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading data...")
    # load_cached_data=True utilizes preprocessed .npy files in ./working
    train_loader, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=2, load_cached_data=False
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = ResNet34UNetPlusPlus(in_channels=2, n_classes=1).to(device)

    # -------------------------------------------------------------------------
    # 4. Training Setup
    # -------------------------------------------------------------------------
    # Using BCEDiceLoss as required
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0, smooth=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_name=CHECKPOINT_NAME,
    )

    # -------------------------------------------------------------------------
    # 5. Execution (Training)
    # -------------------------------------------------------------------------
    best_val_loss = trainer.run()

    # -------------------------------------------------------------------------
    # 6. Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Validation & Failure Analysis ---")

    # Load the best model for analysis
    best_model_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Prepare for metric calculation and failure analysis
    all_preds_cropped = []
    all_masks_cropped = []

    # Padding calculations to crop back to 101x101
    pad_h = TARGET_SIZE - ORIG_SIZE
    pad_top = pad_h // 2
    pad_w = TARGET_SIZE - ORIG_SIZE
    pad_left = pad_w // 2

    with torch.no_grad():
        for inputs, masks in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            outputs = model(inputs)
            preds_prob = torch.sigmoid(outputs).cpu().numpy()
            masks_np = (
                masks.numpy()
            )  # Masks are already on CPU in loader usually, or move back

            # Crop predictions and masks to original size
            # preds_prob shape: (B, 1, 128, 128)
            preds_c = preds_prob[
                ..., pad_top : pad_top + ORIG_SIZE, pad_left : pad_left + ORIG_SIZE
            ]
            masks_c = masks_np[
                ..., pad_top : pad_top + ORIG_SIZE, pad_left : pad_left + ORIG_SIZE
            ]

            all_preds_cropped.append(preds_c)
            all_masks_cropped.append(masks_c)

    # Concatenate all batches
    all_preds_cropped = np.concatenate(all_preds_cropped, axis=0)
    all_masks_cropped = np.concatenate(all_masks_cropped, axis=0)

    # Squeeze channel dimension if necessary (N, 1, H, W) -> (N, H, W)
    if all_preds_cropped.ndim == 4:
        all_preds_cropped = all_preds_cropped.squeeze(1)
    if all_masks_cropped.ndim == 4:
        all_masks_cropped = all_masks_cropped.squeeze(1)

    # --- Calculate Final Validation Metric ---
    final_metric = do_kaggle_metric(all_preds_cropped, all_masks_cropped, threshold=0.5)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # We calculate the Average Precision (AP) per image to define error magnitude.
    # Error Magnitude = 1.0 - AP

    # 1. Calculate IoU per image
    preds_binary = (all_preds_cropped > 0.5).astype(np.uint8)
    masks_binary = all_masks_cropped.astype(np.uint8)

    ious = []
    for i in range(len(preds_binary)):
        p = preds_binary[i]
        t = masks_binary[i]
        intersection = np.sum(p * t)
        union = np.sum(p) + np.sum(t) - intersection
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    ious = np.array(ious)

    # 2. Calculate AP per image (sweeping thresholds 0.5 to 0.95)
    thresholds = np.arange(0.5, 0.95 + 1e-5, 0.05)
    # matches shape: (N_images, N_thresholds)
    matches = ious[:, None] > thresholds[None, :]
    ap_per_image = np.mean(matches, axis=1)

    # 3. Define Error Magnitude
    error_magnitude = 1.0 - ap_per_image

    # 4. Get Input Features (Depth and Coverage)
    # val_loader is not shuffled, so order matches
    depths = val_loader.dataset.depths

    # Calculate coverage from ground truth masks
    coverages = []
    for i in range(len(masks_binary)):
        cov = np.sum(masks_binary[i]) / (ORIG_SIZE * ORIG_SIZE)
        coverages.append(cov)
    coverages = np.array(coverages)

    # 5. Calculate Correlations
    if len(error_magnitude) == len(depths) == len(coverages):
        corr_depth, _ = pearsonr(error_magnitude, depths)
        corr_cov, _ = pearsonr(error_magnitude, coverages)

        print("\nFailure Analysis (Correlation with Error Magnitude):")
        print(f"  Correlation with Depth (z): {corr_depth:.6f}")
        print(f"  Correlation with Salt Coverage: {corr_cov:.6f}")
    else:
        print("Warning: Data length mismatch, skipping correlation analysis.")

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    THRESHOLD_SCORE = 0.7647

    if final_metric > THRESHOLD_SCORE:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )
        # predict_and_submit handles TTA, cropping, and file saving
        predict_and_submit(
            model=model,
            device=device,
            output_dir=SUBMISSION_DIR,
            output_name="submission.csv",
        )
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD_SCORE}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
