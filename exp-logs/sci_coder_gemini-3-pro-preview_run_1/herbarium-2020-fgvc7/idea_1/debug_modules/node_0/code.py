import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Import provided library modules
from library import config
from library import utils
from library import dataset
from library import model
from library import train


def run_demo():
    print("Starting implementation demonstration...")

    # ==========================================
    # 0. Configuration & Patching for Demo Speed
    # ==========================================
    # We patch the constants to ensure the demo runs quickly on a tiny subset
    print("Configuring for fast demonstration...")

    # Patch dataset module's constant because it was imported using 'from ... import ...'
    # This ensures the dataset class uses a tiny subset for this demo run.
    dataset.DEBUG_SAMPLE_SIZE = 20

    # Patch config module's constants
    config.DEBUG = True
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 2

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 1. Verify Utils
    # ==========================================
    print("\n[1/5] Verifying Utils...")

    # Test Reproducibility
    utils.seed_everything(42)
    t1 = torch.rand(5)
    utils.seed_everything(42)
    t2 = torch.rand(5)
    assert torch.equal(
        t1, t2
    ), "Seed everything failed to produce deterministic results."

    # Test AverageMeter
    meter = utils.AverageMeter()
    meter.update(val=5, n=2)  # sum=10, count=2
    meter.update(val=10, n=1)  # sum=20, count=3
    assert abs(meter.avg - 6.666) < 0.01, "AverageMeter calculation incorrect."

    # Test Metrics
    # Simulate a scenario:
    # Class 0: 2 true, 3 predicted (2 correct)
    # Class 1: 2 true, 1 predicted (1 correct)
    # Class 2: 1 true, 1 predicted (1 correct)
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 0])
    f1 = utils.calculate_metrics(y_true, y_pred)
    assert 0.0 <= f1 <= 1.0, "F1 score out of range."
    print("Utils verified successfully.")

    # ==========================================
    # 2. Verify Dataset
    # ==========================================
    print("\n[2/5] Verifying Dataset...")

    # Ensure label mapping is generated
    label2id, id2label = dataset.get_label_mapping(load_cached_data=False)
    assert len(label2id) > 0, "Label mapping is empty."

    # Initialize Train Dataset in Debug mode
    # Note: We patched dataset.DEBUG_SAMPLE_SIZE to 20
    train_ds = dataset.PlantDataset(
        split="train", transform=dataset.get_transforms("train"), debug=True
    )

    assert len(train_ds) <= 20, f"Dataset size {len(train_ds)} exceeds debug limit 20."
    assert len(train_ds) > 0, "Dataset is empty."

    # Check Item
    img, label = train_ds[0]
    assert img.shape == (3, 224, 224), f"Image tensor shape incorrect: {img.shape}"
    assert isinstance(label, torch.Tensor), "Label is not a torch.Tensor"
    assert label.item() < config.NUM_CLASSES, "Label index out of bounds."

    # Check Test Dataset (returns image_id instead of label)
    test_ds = dataset.PlantDataset(
        split="test", transform=dataset.get_transforms("test"), debug=True
    )
    if len(test_ds) > 0:
        t_img, t_id = test_ds[0]
        assert isinstance(
            t_id, (int, np.integer)
        ), f"Test dataset should return image ID, got {type(t_id)}"
    print("Dataset verified successfully.")

    # ==========================================
    # 3. Verify Model
    # ==========================================
    print("\n[3/5] Verifying Model...")

    # Initialize model (pretrained=False to speed up init for this check)
    net = model.PlantClassifier(pretrained=False)
    net.to(device)
    net.eval()

    # Create dummy batch
    dummy_batch = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        outputs = net(dummy_batch)

    assert outputs.shape == (
        2,
        config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {config.NUM_CLASSES}), got {outputs.shape}"
    print("Model verified successfully.")

    # ==========================================
    # 4. Verify Training Loop (Integration)
    # ==========================================
    print("\n[4/5] Verifying Training Pipeline...")

    # We call run_training. We must pass arguments explicitly because defaults were bound at import time.
    # We use the patched config values where possible, but pass num_epochs explicitly.

    print("Running short training cycle...")
    train.run_training(num_epochs=1, batch_size=config.BATCH_SIZE, lr=1e-4, debug=True)

    # Check if best model was saved
    best_model_path = config.CACHE_DIR / "best_model.pth"
    if best_model_path.exists():
        print("Best model checkpoint found.")
    else:
        print(
            "Note: Best model might not be saved if validation F1 didn't improve (possible in very short runs)."
        )

    print("Training pipeline executed.")

    # ==========================================
    # 5. Verify Submission
    # ==========================================
    print("\n[5/5] Verifying Submission...")

    submission_path = config.SUBMISSION_DIR / "submission.csv"
    assert submission_path.exists(), "Submission file not found."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    assert list(sub_df.columns) == ["Id", "Predicted"], "Submission columns mismatch."
    assert len(sub_df) > 0, "Submission file is empty."

    # Check if predictions are valid category IDs
    # The inference function maps label_idx -> category_id using id2label.
    # So all predicted values must be in id2label values.
    valid_cats = set(id2label.values())
    pred_cats = set(sub_df["Predicted"].unique())

    assert pred_cats.issubset(valid_cats), "Submission contains invalid category IDs."

    print("Submission verified successfully.")
    print("\nDone.")


if __name__ == "__main__":
    run_demo()
