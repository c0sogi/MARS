import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, compute_kappa_score
from library.dataset import RetinaDataset, get_transforms
from library.model import RetinaModel, GeM
from library.engine import run, get_ordinal_targets

if __name__ == "__main__":
    # ==========================================
    # 1. Setup & Config Overrides for Demo
    # ==========================================
    print("Setting up configuration for demo run...")

    # Override Config for speed and resource efficiency
    Config.seed = 42
    Config.debug = True  # Limits data to 100 samples per dataset
    Config.epochs = 1  # Run only 1 epoch
    Config.image_size = 224  # Smaller size for faster processing
    Config.batch_size = 8  # Small batch size
    Config.num_workers = 2  # Reduce workers

    # Use a lightweight model and disable pretraining to avoid network dependency/latency
    Config.model_name = "resnet18"
    Config.pretrained = False

    # Redirect working directory to a demo folder
    Config.working_dir = "./working/demo_run"
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    # Update file paths in Config to point to the new working directory
    Config.best_model_path = os.path.join(Config.working_dir, "best_model.pth")
    Config.last_model_path = os.path.join(Config.working_dir, "last_model.pth")
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Ensure reproducibility
    seed_everything(Config.seed)

    # ==========================================
    # 2. Verify Logic: Metrics
    # ==========================================
    print("Verifying Metrics Logic...")
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    score = compute_kappa_score(y_true, y_pred)
    assert score == 1.0, f"Kappa score should be 1.0 for perfect match, got {score}"

    y_pred_bad = [4, 3, 2, 1, 0]
    score_bad = compute_kappa_score(y_true, y_pred_bad)
    assert (
        score_bad < 0
    ), f"Kappa score should be negative for inverse correlation, got {score_bad}"

    # ==========================================
    # 3. Verify Logic: Dataset
    # ==========================================
    print("Verifying Dataset Logic...")
    # Initialize dataset in train mode (using debug subset)
    # Note: load_cached_data=False forces regeneration for this new working_dir
    train_transform = get_transforms(mode="train")
    train_ds = RetinaDataset(
        Config.train_csv,
        mode="train",
        load_cached_data=False,
        transform=train_transform,
    )

    # Check debug subsetting
    assert (
        len(train_ds) <= 100
    ), f"Debug mode should limit dataset size, got {len(train_ds)}"

    # Check item retrieval
    img, label = train_ds[0]
    # Albumentations ToTensorV2 produces [C, H, W]
    expected_shape = (3, Config.image_size, Config.image_size)
    assert (
        img.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {img.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"

    # ==========================================
    # 4. Verify Logic: Model & Pooling
    # ==========================================
    print("Verifying Model Logic...")
    model = RetinaModel().to(Config.device)
    model.eval()

    # Create dummy input [Batch, Channels, Height, Width]
    dummy_input = torch.randn(2, 3, Config.image_size, Config.image_size).to(
        Config.device
    )

    # Verify GeM Pooling independently
    gem = GeM(p=3)
    dummy_feat = torch.randn(2, 64, 16, 16)
    gem_out = gem(dummy_feat)
    # GeM output should be [B, C, 1, 1]
    assert gem_out.shape == (2, 64, 1, 1), f"GeM output shape mismatch: {gem_out.shape}"

    # Verify Full Model Forward Pass
    with torch.no_grad():
        output = model(dummy_input)

    # Output should be [Batch, Num_Classes] -> [2, 4] for Ordinal Regression
    assert output.shape == (2, 4), f"Model output shape mismatch: {output.shape}"

    # ==========================================
    # 5. Verify Logic: Ordinal Targets
    # ==========================================
    print("Verifying Ordinal Target Generation...")
    labels = torch.tensor([0, 2, 4])
    targets = get_ordinal_targets(labels, num_classes=4)
    # Logic:
    # Label 0 -> >0:F, >1:F, >2:F, >3:F -> [0, 0, 0, 0]
    # Label 2 -> >0:T, >1:T, >2:F, >3:F -> [1, 1, 0, 0]
    # Label 4 -> >0:T, >1:T, >2:T, >3:T -> [1, 1, 1, 1]
    expected = torch.tensor(
        [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float
    )

    assert torch.all(targets.cpu() == expected), "Ordinal targets logic incorrect"

    # ==========================================
    # 6. Run Full Pipeline (Training & Inference)
    # ==========================================
    print("Running Full Engine Pipeline...")
    # engine.run() uses the global Config class, which we have modified above.
    # It will train for 1 epoch on the debug subset and run inference.
    run()

    # ==========================================
    # 7. Final Validation
    # ==========================================
    print("Verifying Output Files...")
    assert os.path.exists(Config.submission_path), "Submission file was not created."

    df_sub = pd.read_csv(Config.submission_path)
    assert "id_code" in df_sub.columns, "Submission missing 'id_code' column."
    assert "diagnosis" in df_sub.columns, "Submission missing 'diagnosis' column."
    assert len(df_sub) > 0, "Submission file is empty."

    print("Demo run completed successfully. All checks passed.")
