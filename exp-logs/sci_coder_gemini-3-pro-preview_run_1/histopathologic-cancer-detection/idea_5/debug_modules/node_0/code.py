import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import from the provided library
from library.config import Config
from library.utils import set_seed, calculate_auc, AverageMeter
from library.data import get_dataloaders, PathologyDataset, get_transforms
from library.models import get_model, ModifiedDenseNet, ModifiedResNet
from library.train import run_training, Trainer
from library.inference import run_inference, TTAEngine, load_ensemble_models


def main():
    print("=" * 60)
    print("Running Library Usage Demonstration")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for demo...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce compute load for demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny datasets
    Config.MODEL_NAMES = ["densenet121", "resnet50"]  # Test both ensemble members

    # Setup directories and seeds
    Config.setup()
    print(f"Working directory set to: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Utilities...")

    # Test Seeding
    set_seed(42)
    rand1 = np.random.rand()
    set_seed(42)
    rand2 = np.random.rand()
    assert rand1 == rand2, "Random seed verification failed!"
    print("  - Seeding logic verified.")

    # Test AUC Calculation
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.4, 0.35, 0.8]
    auc = calculate_auc(y_true, y_pred)
    assert 0 <= auc <= 1, "AUC calculation returned invalid range."
    print(f"  - AUC Calculation verified (Score: {auc:.4f}).")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Data Pipeline...")

    # Use the debug flag to load a tiny subset
    loaders = get_dataloaders(
        debug=True,
        sample_size=20,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    assert "train" in loaders and "val" in loaders and "test" in loaders

    # Fetch one batch
    images, labels = next(iter(loaders["train"]))

    # Verify shapes
    # Expected: (Batch, 3, 48, 48) due to Config.CROP_SIZE=48
    assert images.dim() == 4, "Image batch has incorrect dimensions."
    assert images.size(1) == 3, "Image batch should have 3 channels."
    assert (
        images.size(2) == 48 and images.size(3) == 48
    ), f"Image size mismatch. Got {images.shape}"
    assert labels.size(0) == Config.BATCH_SIZE, "Label batch size mismatch."

    print(
        f"  - DataLoader shapes verified: Images {images.shape}, Labels {labels.shape}"
    )

    # -------------------------------------------------------------------------
    # 4. Verify Models
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Model Architecture...")

    for model_name in Config.MODEL_NAMES:
        model = get_model(model_name, pretrained=False)
        model.eval()

        # Check if stem modification was applied (specific to provided library logic)
        if model_name == "densenet121":
            # Original is 7x7 stride 2, Modified is 3x3 stride 1
            conv0 = model.base_model.features.conv0
            assert conv0.kernel_size == (
                3,
                3,
            ), f"{model_name}: Stem kernel size should be 3x3"
            assert conv0.stride == (1, 1), f"{model_name}: Stem stride should be 1"
        elif model_name == "resnet50":
            conv1 = model.base_model.conv1
            assert conv1.kernel_size == (
                3,
                3,
            ), f"{model_name}: Stem kernel size should be 3x3"
            assert conv1.stride == (1, 1), f"{model_name}: Stem stride should be 1"

        # Verify Forward Pass
        with torch.no_grad():
            output = model(images)

        assert output.shape == (
            Config.BATCH_SIZE,
            1,
        ), f"{model_name}: Output shape mismatch. Expected ({Config.BATCH_SIZE}, 1), got {output.shape}"
        print(f"  - {model_name} instantiated and forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Run Training (Demo)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Training Loop (Debug Mode)...")

    # This runs the full training logic provided in library.train
    # It will train both models in the ensemble for 1 epoch on a small subset
    run_training(debug=True, sample_size=30)

    # Verify artifacts were created
    for model_name in Config.MODEL_NAMES:
        expected_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")
        assert os.path.exists(
            expected_path
        ), f"Model checkpoint for {model_name} was not saved."
        print(f"  - Checkpoint found: {expected_path}")

    # -------------------------------------------------------------------------
    # 6. Run Inference (Demo)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running Inference Pipeline (Debug Mode)...")

    # This runs the full inference logic provided in library.inference
    # It loads the models saved in Step 5, runs TTA, and saves submission
    run_inference(debug=True, sample_size=30)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in sub_df.columns and "label" in sub_df.columns
    ), "Submission CSV missing required columns."
    assert len(sub_df) > 0, "Submission CSV is empty."

    print(f"  - Submission generated at {Config.SUBMISSION_PATH}")
    print(f"  - First few rows:\n{sub_df.head()}")

    print("\n" + "=" * 60)
    print("Demonstration Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
