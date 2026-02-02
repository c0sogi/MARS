import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihood, AverageMeter
from library.data import (
    load_dicom_volume,
    generate_tri_slab_view,
    LungDataset,
    get_dataloaders,
)
from library.model import DualAxisTransformer, GeM
from library.train import train_one_epoch, validate


def run_demo():
    print("Initializing Demo...")

    # ==========================================
    # 1. Configuration Setup for Demo
    # ==========================================
    # Override Config defaults for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Reduce computational load
    Config.IMG_SIZE = 128  # Smaller images
    Config.BATCH_SIZE = 4  # Small batch
    Config.EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.PRETRAINED = False  # Avoid downloading weights
    Config.DEBUG = True  # Use data subset
    Config.DEBUG_SAMPLE_SIZE = 10  # Very small subset

    # Setup environment
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Processing Verification
    # ==========================================
    print("\n--- Verifying Data Processing ---")

    # Load metadata to find a valid patient
    train_df = pd.read_csv(Config.TRAIN_CSV)
    sample_patient = train_df.iloc[0]
    patient_id = sample_patient["Patient"]
    dicom_rel_path = sample_patient["dicom_dir"]
    dicom_full_path = os.path.join(Config.INPUT_DIR, dicom_rel_path)

    print(f"Testing DICOM loading for patient: {patient_id}")

    # Test 1: Load Volume
    volume = load_dicom_volume(dicom_full_path, img_size=Config.IMG_SIZE)
    print(f"Volume shape: {volume.shape}")
    assert len(volume.shape) == 3, "Volume must be 3D (D, H, W)"
    assert (
        volume.shape[1] == Config.IMG_SIZE and volume.shape[2] == Config.IMG_SIZE
    ), f"Volume spatial dims should be {Config.IMG_SIZE}"

    # Test 2: Generate Tri-Slab Views
    # Axial View
    img_ax = generate_tri_slab_view(
        volume, axis=0, num_slabs=Config.NUM_SLABS, overlap=Config.SLAB_OVERLAP
    )
    print(f"Axial View shape: {img_ax.shape}, Dtype: {img_ax.dtype}")
    assert img_ax.shape == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
        3,
    ), "Axial view must be (H, W, 3)"
    assert img_ax.dtype == np.uint8, "Image must be uint8"

    # Coronal View
    img_cor = generate_tri_slab_view(
        volume, axis=1, num_slabs=Config.NUM_SLABS, overlap=Config.SLAB_OVERLAP
    )
    print(f"Coronal View shape: {img_cor.shape}")
    assert img_cor.shape == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
        3,
    ), "Coronal view must be (H, W, 3)"

    # ==========================================
    # 3. Dataset & DataLoader Verification
    # ==========================================
    print("\n--- Verifying Dataset & DataLoader ---")

    # Test Dataset instantiation
    ds = LungDataset(train_df.head(10), mode="train", cache_dir=Config.CACHE_DIR)
    item = ds[0]

    required_keys = ["img_ax", "img_cor", "tabular", "meta", "target", "patient_week"]
    for key in required_keys:
        assert key in item, f"Dataset item missing key: {key}"

    print("Dataset item keys verified.")
    print(f"Tabular feature shape: {item['tabular'].shape}")
    assert item["tabular"].shape[0] == 8, "Tabular features should have 8 dimensions"

    # Test DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)
    print(f"Train loader length: {len(train_loader)}")

    # Fetch one batch
    batch = next(iter(train_loader))
    img_ax_batch = batch["img_ax"].to(device)
    img_cor_batch = batch["img_cor"].to(device)
    tabular_batch = batch["tabular"].to(device)
    target_batch = batch["target"].to(device)

    print(
        f"Batch shapes -> Ax: {img_ax_batch.shape}, Cor: {img_cor_batch.shape}, Tab: {tabular_batch.shape}"
    )
    assert img_ax_batch.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert img_ax_batch.shape[1] == 3, "Channel dimension mismatch (should be 3)"

    # ==========================================
    # 4. Model Verification
    # ==========================================
    print("\n--- Verifying Model Architecture ---")

    # Test GeM Pooling
    gem = GeM(p=3)
    dummy_feat = torch.randn(2, 64, 8, 8)
    pooled = gem(dummy_feat)
    assert pooled.shape == (
        2,
        64,
        1,
        1,
    ), f"GeM pooling output shape mismatch: {pooled.shape}"
    print("GeM Pooling verified.")

    # Test DualAxisTransformer
    model = DualAxisTransformer().to(device)

    # Forward Pass
    preds = model(img_ax_batch, img_cor_batch, tabular_batch)
    print(f"Model Output shape: {preds.shape}")

    # Assertions
    assert preds.shape == (Config.BATCH_SIZE, 3), "Output must be (B, 3)"

    # Check positivity of sigma (indices 1 and 2)
    sigmas = preds[:, 1:]
    assert torch.all(sigmas > 0), "Sigma predictions must be positive (Softplus)"
    print("Model forward pass verified.")

    # ==========================================
    # 5. Metric Verification
    # ==========================================
    print("\n--- Verifying Metric (Laplace Log Likelihood) ---")

    # Case 1: Perfect prediction
    y_true = torch.tensor([2000.0]).to(device)
    y_pred = torch.tensor([2000.0]).to(device)
    sigma = torch.tensor([100.0]).to(device)  # > 70, so not clipped

    # Metric = - (sqrt(2) * 0) / 100 - ln(sqrt(2) * 100)
    #        = 0 - ln(141.42) approx -4.95
    score = LaplaceLogLikelihood(y_true, y_pred, sigma)
    expected = -torch.log(torch.tensor(np.sqrt(2) * 100.0)).to(device)

    print(f"Score (Perfect): {score.item():.4f}, Expected: {expected.item():.4f}")
    assert (
        torch.abs(score - expected) < 1e-4
    ), "Metric calculation mismatch for perfect case"

    # Case 2: Clipped Sigma
    sigma_small = torch.tensor([10.0]).to(device)  # Should clip to 70
    score_clipped = LaplaceLogLikelihood(y_true, y_pred, sigma_small)
    expected_clipped = -torch.log(torch.tensor(np.sqrt(2) * 70.0)).to(device)

    print(f"Score (Clipped Sigma): {score_clipped.item():.4f}")
    assert (
        torch.abs(score_clipped - expected_clipped) < 1e-4
    ), "Metric sigma clipping failed"

    # ==========================================
    # 6. Training Loop Verification
    # ==========================================
    print("\n--- Verifying Training Loop ---")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train one epoch
    train_loss = train_one_epoch(1, model, train_loader, optimizer, device)
    print(f"Train Loss (1 epoch): {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validate
    val_score = validate(model, val_loader, device)
    print(f"Validation Score: {val_score:.4f}")
    assert not np.isnan(val_score), "Validation score is NaN"

    # Save Model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint not saved"
    print("Model training and checkpointing verified.")

    # ==========================================
    # 7. Inference Verification
    # ==========================================
    print("\n--- Verifying Inference ---")

    # Load model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    results = []
    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            patient_weeks = batch["patient_week"]

            preds = model(img_ax, img_cor, tabular)

            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            delta_week = meta[:, 0]
            base_fvc = meta[:, 1]

            fvc_pred = base_fvc + alpha * delta_week
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            fvc_pred = fvc_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()

            for i, pw in enumerate(patient_weeks):
                conf = max(sigma_pred[i], 70.0)
                results.append(
                    {"Patient_Week": pw, "FVC": fvc_pred[i], "Confidence": conf}
                )

    submission = pd.DataFrame(results)
    print(f"Generated {len(submission)} predictions.")
    print(submission.head())

    # Verify submission format
    assert "Patient_Week" in submission.columns
    assert "FVC" in submission.columns
    assert "Confidence" in submission.columns
    assert len(submission) > 0

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
