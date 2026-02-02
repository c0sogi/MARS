import os
import torch
import pandas as pd
import shutil
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import (
    get_data,
    DogDataset,
    get_train_transforms,
    get_valid_transforms,
)
from library.model import get_model, freeze_backbone, unfreeze_backbone
from library.engine import train_loop
from library.weight_averaging import average_checkpoints
from library.inference import run_inference


def run_demo():
    print("=== Starting Dog Breed Classification Demo ===\n")

    # ---------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # ---------------------------------------------------------
    print(">>> 1. Configuring Environment")
    seed_everything(42)
    device = get_device()
    print(f"Device selected: {device}")

    # Override Config for fast execution
    Config.WORKING_DIR = "./working/demo_run"
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 16  # Very small subset for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 2  # Minimum epochs to test logic
    Config.PHASE1_EPOCHS = 1
    Config.SWA_EPOCHS = 1
    Config.N_FOLDS = 1  # Run only one fold
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Clean up demo directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"Working directory set to: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Dataset Verification
    # ---------------------------------------------------------
    print("\n>>> 2. Verifying Dataset Pipeline")
    # Load metadata (cached or fresh)
    df_train, class_to_idx, idx_to_class = get_data("train")
    print(f"Train DataFrame Shape (Debug Mode): {df_train.shape}")

    # Assertions
    assert len(df_train) == Config.DEBUG_SUBSET_SIZE, "Debug subset size mismatch"
    assert len(class_to_idx) == Config.NUM_CLASSES, "Class mapping mismatch"

    # Instantiate Dataset
    train_dataset = DogDataset(
        df_train,
        class_to_idx=class_to_idx,
        transforms=get_train_transforms(Config.IMG_SIZE),
    )

    # Verify single item retrieval
    img, label = train_dataset[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label: {label} ({idx_to_class[label.item()]})")

    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image dimensions"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Avoid multiprocessing overhead in demo
    )

    # ---------------------------------------------------------
    # 3. Model Verification
    # ---------------------------------------------------------
    print("\n>>> 3. Verifying Model Architecture")
    # Initialize model (pretrained=False for speed/offline safety)
    model = get_model(pretrained=False)
    model.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"

    # Verify Freeze/Unfreeze Logic
    freeze_backbone(model)
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Frozen Params: {len(frozen_params)}, Trainable Params: {len(trainable_params)}"
    )
    assert len(frozen_params) > 0, "Backbone should be frozen"

    unfreeze_backbone(model)
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    assert len(frozen_params) == 0, "All parameters should be unfrozen"

    # ---------------------------------------------------------
    # 4. Training Loop Execution
    # ---------------------------------------------------------
    print("\n>>> 4. Executing Training Loop (Fold 0)")

    # Prepare Validation Loader
    df_val, _, _ = get_data("val")
    val_dataset = DogDataset(
        df_val,
        class_to_idx=class_to_idx,
        transforms=get_valid_transforms(Config.IMG_SIZE),
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=0)

    # Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Run Loop
    train_loop(
        model, train_loader, val_loader, optimizer, scheduler, device, fold_idx=0
    )

    # Verify Checkpoints
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model_fold_0.pth")
    # SWA saves epoch 1 (since EPOCHS=2, SWA_EPOCHS=1, 2-1=1)
    swa_path = os.path.join(Config.WORKING_DIR, "swa_fold_0_epoch_1.pth")

    assert os.path.exists(best_model_path), "Best model checkpoint missing"
    assert os.path.exists(swa_path), "SWA checkpoint missing"
    print("Training completed and checkpoints verified.")

    # ---------------------------------------------------------
    # 5. Weight Averaging
    # ---------------------------------------------------------
    print("\n>>> 5. Performing Weight Averaging")
    # For demo, we use the generated SWA checkpoint.
    # In a real run, there would be multiple (e.g., epoch_25, epoch_26...).
    checkpoints = [swa_path]
    output_avg_path = os.path.join(Config.WORKING_DIR, "model_fold_0.pth")

    avg_state = average_checkpoints(checkpoints, output_avg_path)
    assert avg_state is not None, "Averaging returned None"
    assert os.path.exists(output_avg_path), "Averaged model file missing"
    print("Weight averaging successful.")

    # ---------------------------------------------------------
    # 6. Inference Pipeline
    # ---------------------------------------------------------
    print("\n>>> 6. Running Inference")
    # run_inference uses Config settings.
    # It will look for 'model_fold_0.pth' because Config.N_FOLDS=1.

    run_inference()

    # Validate Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file missing"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission DataFrame Shape: {sub_df.shape}")

    # Expected shape: (DEBUG_SUBSET_SIZE, 1 + NUM_CLASSES)
    # The '1' is for the 'id' column.
    expected_rows = Config.DEBUG_SUBSET_SIZE
    expected_cols = 1 + Config.NUM_CLASSES

    assert sub_df.shape == (
        expected_rows,
        expected_cols,
    ), f"Submission shape mismatch. Expected ({expected_rows}, {expected_cols}), got {sub_df.shape}"

    # Check if probabilities sum roughly to 1 (sanity check)
    # Select only breed columns (drop 'id')
    probs = sub_df.drop(columns=["id"]).iloc[0].values
    prob_sum = np.sum(probs)
    print(f"Sample probability sum: {prob_sum:.4f}")
    assert np.isclose(prob_sum, 1.0, atol=1e-2), "Probabilities do not sum to 1"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
