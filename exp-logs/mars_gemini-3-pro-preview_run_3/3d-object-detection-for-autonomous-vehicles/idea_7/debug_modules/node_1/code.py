import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
# We import config first to set seeds, though the Engine sets them again.
import library.config as config
import library.utils as utils
import library.dataset as dataset_module
import library.model as model_module
import library.loss as loss_module
import library.engine as engine_module


def test_dataset_logic():
    print("\n=== Testing Dataset Logic ===")

    # 1. Instantiate Dataset
    # We use the existing metadata file.
    metadata_path = os.path.join(config.METADATA_DIR, "train_metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    ds = dataset_module.NuScenesDataset(metadata_path=metadata_path, split="train")
    print(f"Dataset loaded. Total samples: {len(ds)}")

    # 2. Test __getitem__
    sample_idx = 0
    sample = ds[sample_idx]

    # Verify keys
    expected_keys = ["points", "gt_boxes", "gt_labels", "metadata", "targets"]
    for k in expected_keys:
        assert k in sample, f"Missing key {k} in dataset sample"

    # Verify Points: (N, 4) -> x, y, z, intensity
    points = sample["points"]
    assert isinstance(points, np.ndarray), "Points should be numpy array"
    assert points.shape[1] == 4, f"Expected 4 point features, got {points.shape[1]}"

    # Verify Targets
    targets = sample["targets"]
    assert "heatmap" in targets, "Missing heatmap in targets"
    heatmap = targets["heatmap"]
    # Heatmap shape: (Num_Classes, H, W) -> (9, 320, 320) based on config
    assert (
        heatmap.shape[0] == config.NUM_CLASSES
    ), f"Heatmap classes mismatch: {heatmap.shape[0]}"

    print("Dataset __getitem__ verification successful.")

    # 3. Test Collate Function
    batch_list = [ds[0], ds[1]]
    collated = dataset_module.custom_collate_fn(batch_list)

    assert "points" in collated
    assert "targets" in collated
    # Points should have batch index added: (N_total, 5)
    assert (
        collated["points"].shape[1] == 5
    ), "Collated points should have 5 columns (batch_idx + features)"

    # Verify batch stacking of targets
    assert (
        collated["targets"]["heatmap"].shape[0] == 2
    ), "Batch size for heatmap should be 2"

    print("Dataset Collate verification successful.")
    return collated


def test_model_logic(collated_batch):
    print("\n=== Testing Model Logic ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_module.PointPillarsResNetFPN().to(device)
    model.eval()

    points = collated_batch["points"].to(device)

    # Forward pass
    with torch.no_grad():
        preds = model(points)

    # Verify Output Heads
    expected_heads = ["heatmap", "reg", "height", "dim", "rot"]
    for head in expected_heads:
        assert head in preds, f"Model output missing head: {head}"

    # Check shape of heatmap: (B, C, H, W)
    # B=2, C=9, H=320, W=320
    heatmap = preds["heatmap"]
    assert heatmap.shape[0] == 2
    assert heatmap.shape[1] == config.NUM_CLASSES

    print("Model forward pass verification successful.")
    return model, preds


def test_loss_logic(preds, collated_batch):
    print("\n=== Testing Loss Logic ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = loss_module.CenterLoss()

    # Move targets to device
    targets = {k: v.to(device) for k, v in collated_batch["targets"].items()}

    # Calculate Loss
    loss, loss_stats = criterion(preds, targets)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print(f"Loss calculation successful. Value: {loss.item():.4f}")
    print("Loss Stats:", loss_stats)


def run_engine_demo():
    print("\n=== Running Engine Demo (Training & Inference) ===")

    # Patch Engine Configuration for Speed
    # The engine module imports constants from config. We need to patch them in the engine module namespace.
    engine_module.TRAIN_SUBSET_SIZE = 20  # Use only 20 samples for training
    engine_module.VAL_SUBSET_SIZE = 10  # Use only 10 samples for validation
    engine_module.BATCH_SIZE = 4
    engine_module.NUM_EPOCHS = 1  # Run only 1 epoch
    engine_module.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Instantiate Engine
    engine = engine_module.Engine()

    # Run Training
    print("Starting training loop...")
    engine.run_training()

    # Verify Checkpoint creation
    checkpoint_path = os.path.join(config.WORKING_DIR, "model_checkpoint.pth")
    assert os.path.exists(checkpoint_path), "Model checkpoint was not created."
    print("Training loop completed and checkpoint verified.")

    # Run Submission Generation
    # We also need to patch the test dataset loading inside generate_submission if we want it fast,
    # but the method loads the full test set.
    # To speed this up for the demo, we will temporarily mock the test metadata file
    # to contain only a few rows, or just let it run (test set is ~4.5k samples, inference is fast).
    # Given the constraint "Optimize for Speed", let's create a subset metadata file for testing.

    original_test_meta_path = os.path.join(config.METADATA_DIR, "test_metadata.csv")
    subset_test_meta_path = os.path.join(
        config.METADATA_DIR, "test_metadata_subset.csv"
    )

    if os.path.exists(original_test_meta_path):
        df = pd.read_csv(original_test_meta_path)
        df_subset = df.head(10)  # Only 10 test samples
        df_subset.to_csv(subset_test_meta_path, index=False)

        # Monkey patch the dataset class temporarily to use the subset file when 'test' split is requested
        # However, the engine hardcodes the path.
        # We will monkey patch the Engine.generate_submission method's internal path usage? No, that's hard.
        # Easier: The Engine.generate_submission method instantiates NuScenesDataset with specific path.
        # We can just run it. 4500 samples inference might take 1-2 mins on GPU. That is acceptable.
        pass

    print("Generating submission...")
    engine.generate_submission()

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."
    print("Submission generation successful.")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Verify Dataset
    batch_data = test_dataset_logic()

    # 2. Verify Model
    model, predictions = test_model_logic(batch_data)

    # 3. Verify Loss
    test_loss_logic(predictions, batch_data)

    # 4. Run Engine (Train + Inference)
    run_engine_demo()

    print("\nAll verification steps completed successfully.")
