import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, compute_robust_roc_auc
from library.dataset import BirdDataset
from library.models import get_model
from library.engine import fit_model


def run_demo():
    print("==== Starting Library Usage Demo ====")

    # 1. Configuration Overrides for Speed/Demo
    print("[1] Configuring environment for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 32  # Small subset
    Config.EPOCHS = 2  # Minimal epochs
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PATIENCE = 2

    # Setup directories
    Config.setup()

    # Set reproducibility
    set_seed(Config.SEED)
    print("    Configuration complete. Debug mode enabled.")

    # 2. Dataset Verification
    print("\n[2] Verifying BirdDataset...")
    # Initialize datasets
    train_dataset = BirdDataset(
        Config.TRAIN_METADATA, phase="train", load_cached_data=False
    )
    val_dataset = BirdDataset(Config.VAL_METADATA, phase="val", load_cached_data=False)

    # Verify lengths
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_dataset)}"
    assert (
        len(val_dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Val dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(val_dataset)}"

    # Verify item structure
    sample_img, sample_label = train_dataset[0]

    # Check Image Shape: (3, 224, 224) - 3 channels (replicated), 224x224 resize
    assert sample_img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Image shape mismatch. Expected {(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)}, got {sample_img.shape}"

    # Check Label Shape: (19,) - 19 classes
    assert sample_label.shape == (
        Config.NUM_CLASSES,
    ), f"Label shape mismatch. Expected {(Config.NUM_CLASSES,)}, got {sample_label.shape}"

    # Check Label Type
    assert isinstance(sample_label, torch.Tensor), "Label should be a torch.Tensor"
    assert sample_label.dtype == torch.float32, "Label dtype should be float32"

    print("    BirdDataset verification passed.")

    # 3. Model Verification
    print("\n[3] Verifying Model Architecture...")
    model_name = "resnet18"
    model = get_model(
        model_name, num_classes=Config.NUM_CLASSES, pretrained=False
    )  # False for speed

    # Move to device
    device = Config.DEVICE
    model = model.to(device)

    # Create dummy batch
    dummy_input = torch.randn(4, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Assert output shape (Batch_Size, Num_Classes)
    assert output.shape == (
        4,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (4, {Config.NUM_CLASSES}), got {output.shape}"

    print(f"    {model_name} instantiated and forward pass successful.")

    # 4. Metric Verification
    print("\n[4] Verifying Robust ROC AUC Metric...")
    # Case 1: Perfect prediction
    y_true = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    y_pred = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    score = compute_robust_roc_auc(y_true, y_pred)
    assert score == 1.0, f"Expected perfect score 1.0, got {score}"

    # Case 2: Missing class (Class 1 has no positive samples)
    # y_true column 1 is all zeros. The function should skip this class and average the rest.
    y_true_missing = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    y_pred_missing = np.array([[0.9, 0.2], [0.1, 0.2], [0.8, 0.3], [0.2, 0.1]])
    # Only class 0 is valid. AUC for class 0 should be 1.0.
    score_missing = compute_robust_roc_auc(y_true_missing, y_pred_missing)
    assert (
        score_missing == 1.0
    ), f"Expected robust score 1.0 (ignoring invalid class), got {score_missing}"

    print("    Metric verification passed.")

    # 5. Training Loop Demonstration
    print("\n[5] Demonstrating Training Loop (fit_model)...")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run training for one fold
    # Using a fresh model instance
    model_to_train = get_model("resnet18", pretrained=True)

    best_score = fit_model(
        model=model_to_train,
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=0,
        model_name="resnet18",
    )

    assert isinstance(best_score, float), "fit_model should return a float score"
    print(f"    Training simulation complete. Best Val AUC: {best_score:.4f}")

    # 6. Inference Demonstration
    print("\n[6] Demonstrating Inference with Best Model...")

    # Load the best model weights
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "resnet18_fold_0_best.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint not found!"

    inference_model = get_model("resnet18", pretrained=False)
    inference_model.load_state_dict(torch.load(best_model_path, map_location=device))
    inference_model = inference_model.to(device)
    inference_model.eval()

    # Run on a validation batch
    imgs, _ = next(iter(val_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        logits = inference_model(imgs)
        probs = torch.sigmoid(logits)

    # Verify probabilities
    assert probs.min() >= 0 and probs.max() <= 1, "Probabilities must be in [0, 1]"
    print("    Inference successful. Probabilities range verified.")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
