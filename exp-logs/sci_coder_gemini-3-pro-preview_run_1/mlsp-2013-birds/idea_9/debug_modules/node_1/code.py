import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import provided library modules
from library import config
from library import utils
from library import data
from library import model
from library import engine
from library import pipeline


def test_utils():
    print("\n=== Testing Library: Utils ===")

    # Test Seeding
    utils.set_seed(42)
    r1 = np.random.rand()
    utils.set_seed(42)
    r2 = np.random.rand()
    assert r1 == r2, "Seeding failed: Random numbers are not reproducible."
    print("Seeding verification passed.")

    # Test AverageMeter
    meter = utils.AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=1)  # Sum = 20 + 20 = 40, Count = 3
    assert meter.avg == 40 / 3, f"AverageMeter failed: Expected {40/3}, got {meter.avg}"
    print("AverageMeter verification passed.")

    # Test ROC AUC Computation
    # Case 1: Perfect prediction
    y_true = np.array([[0, 1], [1, 0], [0, 1]])
    y_pred = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8]])
    score = utils.compute_roc_auc(y_true, y_pred)
    assert score == 1.0, f"ROC AUC failed: Expected 1.0, got {score}"

    # Case 2: Missing class in batch (should be handled robustly)
    y_true_missing = np.array([[0, 1], [0, 1]])  # Class 0 is always 0
    # The function returns mean of calculable classes. Here Class 1 is constant too?
    # Actually if a class has only one label (all 0 or all 1), sklearn raises error.
    # The provided util catches this and returns 0.0 if no classes are valid,
    # or mean of valid ones.
    # Let's try a case where one class is valid and one is not.
    y_true_mixed = np.array([[0, 1], [1, 1], [0, 1]])
    # Col 0: [0, 1, 0] -> Valid
    # Col 1: [1, 1, 1] -> Invalid (only one unique label)
    y_pred_mixed = np.array([[0.1, 0.9], [0.9, 0.9], [0.2, 0.9]])
    score_mixed = utils.compute_roc_auc(y_true_mixed, y_pred_mixed)
    # Only col 0 is computed. AUC for col 0 is 1.0.
    assert score_mixed == 1.0, f"ROC AUC robust handling failed: Got {score_mixed}"
    print("ROC AUC computation verification passed.")


def test_data_processing():
    print("\n=== Testing Library: Data ===")

    # Test Metadata Loading
    df_train = data.load_metadata("train")
    assert isinstance(df_train, pd.DataFrame), "Failed to load train metadata"
    assert len(df_train) > 0, "Train metadata is empty"
    print(f"Metadata loaded. Train size: {len(df_train)}")

    # Test Data Processing and Caching
    # We use a small subset for speed
    df_subset = df_train.head(10)
    width, height = 128, 128

    print("Processing subset of data...")
    images, labels, ids = data.process_and_cache_data(
        df_subset, "demo_test", width, height, load_cached_data=False
    )

    # Verify shapes
    assert isinstance(images, torch.Tensor)
    assert images.shape == (
        10,
        3,
        height,
        width,
    ), f"Image shape mismatch: {images.shape}"
    assert labels.shape == (
        10,
        config.NUM_CLASSES,
    ), f"Label shape mismatch: {labels.shape}"
    assert len(ids) == 10

    # Verify Normalization (roughly)
    # ImageNet mean is approx 0.45, std 0.22. Values should be centered around 0.
    # Since inputs are spectrograms (often black background), values might be lower,
    # but shouldn't be 0-255.
    assert (
        images.max() <= 10.0 and images.min() >= -10.0
    ), "Images do not appear normalized."
    print("Data processing verification passed.")

    # Test DataLoader
    loader = data.get_loader(images, labels, ids, batch_size=4, shuffle=False)
    batch_images, batch_labels, batch_ids = next(iter(loader))
    assert batch_images.shape == (4, 3, height, width)
    assert batch_labels.shape == (4, config.NUM_CLASSES)
    print("DataLoader verification passed.")

    # Test Mixup
    mixed_x, y_a, y_b, lam = data.mixup_data(
        batch_images, batch_labels, alpha=1.0, device="cpu"
    )
    assert mixed_x.shape == batch_images.shape
    assert y_a.shape == batch_labels.shape
    assert 0 <= lam <= 1
    print("Mixup verification passed.")


def test_model_architecture():
    print("\n=== Testing Library: Model ===")

    # Initialize model
    net = model.get_bird_model(pretrained=False)
    net.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, 256, 256)

    # Forward pass
    with torch.no_grad():
        output = net(dummy_input)

    # Verify output shape
    assert output.shape == (
        2,
        config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {config.NUM_CLASSES}), got {output.shape}"
    print("Model architecture verification passed.")


def test_engine():
    print("\n=== Testing Library: Engine ===")

    device = "cpu"  # Force CPU for simple test to avoid CUDA initialization overhead if not needed

    # Setup dummy data and model
    net = model.get_bird_model(pretrained=False).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Create dummy dataset
    images = torch.randn(8, 3, 128, 128)
    labels = torch.randint(0, 2, (8, config.NUM_CLASSES)).float()
    ids = np.arange(8)
    loader = data.get_loader(images, labels, ids, batch_size=4, shuffle=True)

    # Test Train Step
    loss = engine.train_one_epoch(
        net, loader, optimizer, device, epoch=0, mixup_alpha=0.0
    )
    assert isinstance(loss, float)
    assert loss > 0
    print(f"Engine training step passed. Loss: {loss:.4f}")

    # Test Evaluation Step
    val_loss, val_auc, probs, targets = engine.evaluate(net, loader, device)
    assert isinstance(val_loss, float)
    assert 0.0 <= val_auc <= 1.0
    assert probs.shape == (8, config.NUM_CLASSES)
    print(f"Engine evaluation step passed. AUC: {val_auc:.4f}")


def test_full_pipeline():
    print("\n=== Testing Library: Full Pipeline (Debug Mode) ===")

    # Clean working directory to ensure fresh run
    if os.path.exists(config.WORKING_DIR):
        # We don't delete the whole dir as it might contain other things,
        # but the pipeline functions overwrite files so it's fine.
        pass

    # 1. Train Teachers
    # debug=True runs for fewer epochs and on a subset of data
    print(">> Running Stage 1: Train Teachers")
    teacher_paths = pipeline.train_teachers(debug=True)

    assert isinstance(teacher_paths, list)
    assert len(teacher_paths) == len(config.TEACHER_WIDTHS)
    for path in teacher_paths:
        assert os.path.exists(path), f"Teacher model not saved at {path}"
    print("Stage 1 completed successfully.")

    # 2. Generate Pseudo Labels
    print(">> Running Stage 2: Generate Pseudo Labels")
    test_ids, pseudo_labels = pipeline.generate_ensemble_pseudo_labels(
        teacher_paths, debug=True
    )

    assert test_ids is not None
    assert pseudo_labels is not None
    # In debug mode, subset size is 20
    assert len(test_ids) == 20
    assert pseudo_labels.shape == (20, config.NUM_CLASSES)
    print("Stage 2 completed successfully.")

    # 3. Train Student with SWA
    print(">> Running Stage 3: Train Student with SWA")
    student_model = pipeline.train_student_with_swa(test_ids, pseudo_labels, debug=True)

    assert isinstance(student_model, torch.nn.Module)
    student_path = os.path.join(config.WORKING_DIR, "student_swa.pth")
    assert os.path.exists(student_path), "Student model checkpoint not found."
    print("Stage 3 completed successfully.")


def main():
    # Ensure reproducibility
    utils.set_seed(42)

    # Run tests
    test_utils()
    test_data_processing()
    test_model_architecture()
    test_engine()
    test_full_pipeline()

    print("\nALL TESTS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
