import os
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import OSICDataset
from library.model import SCSLNet
from library.engine import train_one_epoch, evaluate, LaplaceLoss


def main():
    print("=== Starting SCSL-Net Demo Script ===")

    # 1. Configuration & Setup
    # Override Config for a fast demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.CACHE_DIR = "./working/demo_cache"  # Use a temp cache for this run

    # Ensure clean state
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Preparation
    print("\n[Step 1] Loading Metadata and Preparing Subsets...")
    train_csv_path = Config.TRAIN_CSV
    test_csv_path = Config.TEST_CSV

    # Load metadata
    df_train_full = pd.read_csv(train_csv_path)
    df_test_full = pd.read_csv(test_csv_path)

    # Select a small subset of patients for the demo to speed up DICOM processing
    # We pick 2 unique patients for training
    train_patients = df_train_full["Patient"].unique()[:2]
    df_train_subset = df_train_full[
        df_train_full["Patient"].isin(train_patients)
    ].copy()

    # We pick 1 unique patient for testing
    test_patients = df_test_full["Patient"].unique()[:1]
    df_test_subset = df_test_full[df_test_full["Patient"].isin(test_patients)].copy()

    print(
        f"Training Subset: {len(df_train_subset)} rows ({len(train_patients)} patients)"
    )
    print(f"Test Subset: {len(df_test_subset)} rows ({len(test_patients)} patients)")

    # 3. Dataset & DataLoader Instantiation
    print("\n[Step 2] Initializing OSICDataset...")
    # Initialize dataset
    train_dataset = OSICDataset(
        df_train_subset,
        mode="train",
        transform=None,  # Use default ToTensorV2 logic in __getitem__ if None
        load_cached_data=False,  # Force processing for demonstration
    )

    # Verify __getitem__
    sample = train_dataset[0]
    print("Sample keys:", sample.keys())

    # Assertions to verify data shapes
    # Images should be (3, 224, 224)
    assert sample["image_axial"].shape == (
        3,
        224,
        224,
    ), f"Axial shape mismatch: {sample['image_axial'].shape}"
    assert sample["image_coronal"].shape == (
        3,
        224,
        224,
    ), f"Coronal shape mismatch: {sample['image_coronal'].shape}"
    # Tabular should be (4,) -> Age, Sex, Smoke, Percent
    assert sample["tabular"].shape == (
        4,
    ), f"Tabular shape mismatch: {sample['tabular'].shape}"
    # Target should be scalar
    assert sample["target"].ndim == 0, "Target should be scalar"

    print("Dataset verification successful.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    # 4. Model Initialization
    print("\n[Step 3] Initializing SCSLNet Model...")
    model = SCSLNet().to(device)

    # Verify Forward Pass
    dummy_batch = next(iter(train_loader))
    img_ax = dummy_batch["image_axial"].to(device)
    img_cor = dummy_batch["image_coronal"].to(device)
    tab = dummy_batch["tabular"].to(device)

    with torch.no_grad():
        preds = model(img_ax, img_cor, tab)

    print(f"Prediction Shape: {preds.shape}")
    # Output should be (Batch, 3) -> [alpha, sigma_base, sigma_growth]
    assert preds.shape == (Config.BATCH_SIZE, 3), "Model output shape incorrect."
    print("Model forward pass verification successful.")

    # 5. Training Loop Demonstration
    print("\n[Step 4] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run training
    avg_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Epoch 1 Loss: {avg_loss:.4f}")

    # Verify loss is a valid float
    assert not np.isnan(avg_loss), "Training loss resulted in NaN."

    # 6. Evaluation Demonstration
    print("\n[Step 5] Running Evaluation...")
    # Using the same subset as validation for demo purposes
    val_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    metric_score = evaluate(model, val_loader, device)
    print(f"Validation Metric Score: {metric_score:.4f}")
    assert isinstance(metric_score, float), "Metric score should be a float."

    # 7. Inference & Submission Logic
    print("\n[Step 6] Generating Predictions for Test Set...")
    test_dataset = OSICDataset(df_test_subset, mode="test", load_cached_data=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model.eval()
    submission_rows = []

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tab = batch["tabular"].to(device)

            # Metadata for parametric inference
            # In test mode, we need: Patient_Week ID, Baseline FVC, Baseline Week, Predict Week
            # The dataset 'meta' dict provides tensors, we need to map them back to IDs if batching
            # However, the dataset returns 'Patient' string in meta.

            # Note: DataLoader collates strings into list/tuple
            patient_ids = batch["meta"]["Patient"]
            weeks = batch["meta"]["Weeks"].to(
                device
            )  # This is the target prediction week
            base_weeks = batch["meta"]["Baseline_Week"].to(device)
            base_fvc = batch["meta"]["Baseline_FVC"].to(device)

            # Forward
            preds = model(img_ax, img_cor, tab)

            # Parametric decoding
            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            dt = weeks - base_weeks

            # Calculate FVC and Confidence
            fvc_pred = base_fvc + alpha * dt
            sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

            # Move to CPU
            fvc_pred = fvc_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()
            weeks_cpu = weeks.cpu().numpy()

            for i in range(len(patient_ids)):
                pid = patient_ids[i]
                wk = int(weeks_cpu[i])
                fvc = fvc_pred[i]
                conf = sigma_pred[i]

                patient_week = f"{pid}_{wk}"
                submission_rows.append(
                    {"Patient_Week": patient_week, "FVC": fvc, "Confidence": conf}
                )

    # Create submission DataFrame
    sub_df = pd.DataFrame(submission_rows)
    print(f"Generated {len(sub_df)} prediction rows.")
    print(sub_df.head())

    # Save to working directory
    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    # 8. Metric Logic Verification
    print("\n[Step 7] Verifying Metric Logic...")
    # Test case 1: Perfect prediction
    y_true = np.array([2000.0])
    y_pred = np.array([2000.0])
    sigma = np.array([100.0])  # > 70, so clipped is 100

    # Metric = - (sqrt(2)*0)/100 - ln(sqrt(2)*100) = - ln(141.42) approx -4.95
    score_perfect = laplace_log_likelihood_metric(y_true, y_pred, sigma)
    expected_perfect = -np.log(np.sqrt(2) * 100)
    print(f"Score (Perfect): {score_perfect:.4f}, Expected: {expected_perfect:.4f}")
    assert np.isclose(score_perfect, expected_perfect, atol=1e-4)

    # Test case 2: Large error (clipped at 1000)
    y_true_bad = np.array([2000.0])
    y_pred_bad = np.array([4000.0])  # Delta 2000 -> Clipped to 1000
    sigma_bad = np.array([50.0])  # Clipped to 70

    # Metric = - (sqrt(2)*1000)/70 - ln(sqrt(2)*70)
    score_bad = laplace_log_likelihood_metric(y_true_bad, y_pred_bad, sigma_bad)
    expected_bad = -(np.sqrt(2) * 1000) / 70 - np.log(np.sqrt(2) * 70)
    print(f"Score (Bad): {score_bad:.4f}, Expected: {expected_bad:.4f}")
    assert np.isclose(score_bad, expected_bad, atol=1e-4)

    print("Metric logic verification successful.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
