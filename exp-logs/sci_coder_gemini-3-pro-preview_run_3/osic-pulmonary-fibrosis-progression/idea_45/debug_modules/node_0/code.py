import os
import shutil
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import BPCDSNet
from library.loss import LaplaceNLLLoss
from library.engine import train_one_epoch, evaluate


def main():
    print("=== Starting Demonstration of Pulmonary Fibrosis Prediction Pipeline ===")

    # 1. Setup Configuration for Demo
    # We override the Config class attributes to create an isolated, fast execution environment.
    DEMO_DIR = "./working/demo_task_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)

    print(f"Setting up working directory: {DEMO_DIR}")
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Optimization for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 12  # Small subset of patients
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 2
    Config.NUM_SLICES = 3
    Config.IMG_SIZE = 260

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    seed_everything(Config.SEED)

    # 2. Data Preparation and Loading
    print("\n--- Step 1: Data Preparation & Loading ---")
    # get_dataloaders handles image caching and tabular preprocessing internally
    train_loader, val_loader, test_loader, target_stats = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=2,
        load_cached_data=False,  # Force processing for demo
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Target Stats (for inverse scaling): {target_stats}")

    # Verify Data Integrity
    images, clinical, targets = next(iter(train_loader))

    print(f"\nSample Batch Shapes:")
    print(
        f"  Images: {images.shape} (Expected: [{Config.BATCH_SIZE}, {Config.NUM_SLICES}, {Config.IMG_SIZE}, {Config.IMG_SIZE}])"
    )
    print(f"  Clinical: {clinical.shape} (Expected: [{Config.BATCH_SIZE}, 5])")
    print(f"  Targets: {targets.shape} (Expected: [{Config.BATCH_SIZE}, 1])")

    assert images.shape == (
        Config.BATCH_SIZE,
        Config.NUM_SLICES,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image tensor shape"
    assert clinical.shape == (Config.BATCH_SIZE, 5), "Incorrect clinical features shape"
    assert targets.shape == (Config.BATCH_SIZE, 1), "Incorrect target shape"

    # 3. Model Initialization
    print("\n--- Step 2: Model Initialization ---")
    model = BPCDSNet().to(device)

    # Verify Forward Pass
    images = images.to(device)
    clinical = clinical.to(device)

    with torch.no_grad():
        mu, sigma = model(images, clinical)

    print(f"Model Output Shapes:")
    print(f"  Mu (FVC): {mu.shape}")
    print(f"  Sigma (Conf): {sigma.shape}")

    assert mu.shape == (Config.BATCH_SIZE,), "Mu output shape mismatch"
    assert sigma.shape == (Config.BATCH_SIZE,), "Sigma output shape mismatch"
    assert torch.all(sigma > 0), "Sigma must be positive"
    print("Forward pass verification successful.")

    # 4. Training Loop
    print("\n--- Step 3: Training Loop ---")
    loss_fn = LaplaceNLLLoss()

    # Differential Learning Rates as per Config
    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": Config.LR_BACKBONE},
            {"params": model.img_projector.parameters(), "lr": Config.LR_HEADS},
            {"params": model.stream_a.parameters(), "lr": Config.LR_HEADS},
            {"params": model.stream_b.parameters(), "lr": Config.LR_HEADS},
            {"params": model.head.parameters(), "lr": Config.LR_HEADS},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    best_metric = -float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, loss_fn, device
        )
        print(f"  Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_metric = evaluate(
            model, val_loader, loss_fn, device, target_stats
        )
        print(f"  Val Loss: {val_loss:.4f} | Val Metric: {val_metric:.4f}")

        scheduler.step()

        # Checkpoint
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )
            print("  -> Saved Best Model")

    # 5. Inference on Test Set
    print("\n--- Step 4: Inference Simulation ---")
    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
    )
    model.eval()

    predictions = []

    global_mean = target_stats["mean"]
    global_std = target_stats["std"]

    print("Generating predictions for test set...")
    with torch.no_grad():
        for i, (images, clinical) in enumerate(test_loader):
            images = images.to(device)
            clinical = clinical.to(device)

            mu, sigma = model(images, clinical)

            # Inverse Transform
            # mu is Z-scored FVC -> convert to ml
            pred_fvc_ml = mu.cpu().numpy() * global_std + global_mean

            # sigma is scaled confidence -> convert to ml
            pred_sigma_ml = sigma.cpu().numpy() * global_std

            # In a real submission, we would map these back to Patient_Week IDs
            # Here we just verify the values are in a reasonable range (e.g., lung capacity > 0)
            for fvc, conf in zip(pred_fvc_ml, pred_sigma_ml):
                predictions.append((fvc, conf))

    # Basic sanity check on predictions
    if predictions:
        sample_fvc, sample_conf = predictions[0]
        print(
            f"Sample Prediction -> FVC: {sample_fvc:.2f} ml, Confidence: {sample_conf:.2f} ml"
        )
        assert sample_fvc > 0, "Predicted FVC should be positive"
        assert sample_conf > 0, "Predicted Confidence should be positive"

    # 6. Submission File Generation (Mock)
    print("\n--- Step 5: Submission Generation ---")
    # We need to map predictions back to the test dataframe structure
    # The test_loader iterates over the test_df created in prepare_data
    # We reload the processed test dataframe to get IDs
    test_df_path = os.path.join(Config.CACHE_DIR, "test_processed.parquet")
    test_df = pd.read_parquet(test_df_path)

    # Note: The test_loader order matches test_df rows because shuffle=False
    # Ensure lengths match
    if len(predictions) == len(test_df):
        test_df["FVC_pred"] = [p[0] for p in predictions]
        test_df["Confidence_pred"] = [p[1] for p in predictions]

        # Create Patient_Week column
        test_df["Patient_Week"] = (
            test_df["Patient"] + "_" + test_df["Weeks"].astype(str)
        )

        submission = test_df[["Patient_Week", "FVC_pred", "Confidence_pred"]].rename(
            columns={"FVC_pred": "FVC", "Confidence_pred": "Confidence"}
        )

        out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(out_path, index=False)
        print(f"Submission saved to {out_path}")
        print(submission.head())
    else:
        print(
            "Warning: Prediction count mismatch (likely due to DropLast or similar loader settings, though test loader shouldn't drop)."
        )

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
