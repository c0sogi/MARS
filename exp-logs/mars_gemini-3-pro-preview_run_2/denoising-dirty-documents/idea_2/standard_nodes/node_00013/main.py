import os
import sys
import numpy as np
import torch

# Ensure the current directory is in the path to import library modules correctly
sys.path.append(os.getcwd())

from library.utils import set_seed
from library.dataset import get_dataloaders
from library.engine import train_engine
from library.model import ResUNet, predict_tiled
from library.inference import run_inference


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    SEED = 42
    set_seed(SEED)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_PATH = os.path.join(WORKING_DIR, "resunet_best.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for Fast Baseline
    BATCH_SIZE = 16
    PATCH_SIZE = 128
    PATCHES_PER_IMAGE = 8  # Extract 8 patches per image per epoch
    EPOCHS = 20  # Sufficient for convergence on this dataset size
    LR = 1e-4
    PATIENCE = 5

    # ==========================================
    # 2. Data Loading
    # ==========================================
    train_loader, val_data, test_data = get_dataloaders(
        metadata_dir=METADATA_DIR,
        cache_dir=CACHE_DIR,
        input_dir=INPUT_DIR,
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        patches_per_image=PATCHES_PER_IMAGE,
        num_workers=2,
        load_cached=True,
        seed=SEED,
    )

    # ==========================================
    # 3. Training
    # ==========================================
    # train_engine handles the training loop, validation per epoch, and saving the best model
    _ = train_engine(
        train_loader=train_loader,
        val_data=val_data,
        epochs=EPOCHS,
        lr=LR,
        device=DEVICE,
        save_path=MODEL_PATH,
        patience=PATIENCE,
        seed=SEED,
    )

    # ==========================================
    # 4. Final Validation & Failure Analysis
    # ==========================================

    # Load the best saved model
    model = ResUNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        # Fallback if training failed to save (should not happen)
        pass

    model.eval()

    val_sse = 0.0
    total_pixels = 0

    all_errors_abs = []
    all_intensities = []

    # Iterate through validation data for evaluation and analysis
    for item in val_data:
        noisy_np = item["noisy"]
        clean_np = item["clean"]

        # Prepare tensor (1, 1, H, W)
        noisy_t = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).float()
        clean_t = (
            torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        )

        with torch.no_grad():
            # Use tiled inference to handle full resolution
            pred_clean = predict_tiled(model, noisy_t.squeeze(0), device=DEVICE)
            pred_clean = pred_clean.unsqueeze(0)  # Add batch dim back

        # Calculate Squared Error for RMSE
        diff = pred_clean - clean_t
        diff_sq = diff**2
        val_sse += diff_sq.sum().item()
        total_pixels += diff.numel()

        # Collect data for failure analysis: Error Magnitude vs Input Intensity
        err_abs = torch.abs(diff).cpu().numpy().flatten()
        intensity = noisy_np.flatten()

        all_errors_abs.append(err_abs)
        all_intensities.append(intensity)

    # Compute Final RMSE
    final_rmse = np.sqrt(val_sse / total_pixels)

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {final_rmse}")

    # Compute Correlation for Failure Analysis
    flat_errors = np.concatenate(all_errors_abs)
    flat_intensities = np.concatenate(all_intensities)

    if len(flat_errors) > 0:
        # np.corrcoef returns correlation matrix
        corr_matrix = np.corrcoef(flat_errors, flat_intensities)
        correlation = corr_matrix[0, 1]
        print(f"Correlation between Error Magnitude and Input Intensity: {correlation}")
    else:
        print("Correlation between Error Magnitude and Input Intensity: N/A")

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD = 0.055280789732933044

    if final_rmse < THRESHOLD:
        # Ensure submission directory exists
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        run_inference(
            test_data=test_data,
            model_path=MODEL_PATH,
            output_path=SUBMISSION_PATH,
            device=DEVICE,
            patch_size=PATCH_SIZE,
            overlap=32,
            use_tta=True,  # Enable Test Time Augmentation for best performance
        )


if __name__ == "__main__":
    main()
