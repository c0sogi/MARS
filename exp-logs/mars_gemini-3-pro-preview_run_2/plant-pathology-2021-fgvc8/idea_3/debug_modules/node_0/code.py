import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from the provided library files
from library.config import CFG
from library.utils import seed_everything, get_score
from library.data import get_loaders
from library.model import AppleDiseaseModel, GeM
from library.engine import fit_model, train_one_epoch, valid_one_epoch
from library.inference import generate_submission, soft_voting_ensemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def demo_utils():
    """Demonstrate and verify utility functions."""
    print("\n=== 1. Verifying Utility Functions ===")

    # Test get_score (Macro F1)
    # Case 1: Perfect match
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array(
        [[1, 0, 1], [0, 1, 0]]
    )  # Logits/probs don't matter if we pass binary here for logic check,
    # but function expects probs/logits if float.
    # Let's pass float probs that result in these binaries.
    y_pred_probs = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3]])

    score = get_score(y_true, y_pred_probs, threshold=0.5)
    print(f"Perfect Match Score: {score}")
    assert score == 1.0, "F1 Score should be 1.0 for perfect predictions"

    # Case 2: Complete mismatch
    y_pred_bad = np.array([[0.1, 0.9, 0.2], [0.8, 0.2, 0.9]])
    score_bad = get_score(y_true, y_pred_bad, threshold=0.5)
    print(f"Mismatch Score: {score_bad}")
    assert score_bad == 0.0, "F1 Score should be 0.0 for complete mismatch"

    print("Utils verification passed.")


def demo_model_components():
    """Demonstrate and verify model components independent of data."""
    print("\n=== 2. Verifying Model Components ===")

    # Test GeM Pooling
    # Input: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(2, 64, 16, 16)
    gem_layer = GeM(p=3)
    output = gem_layer(dummy_input)

    print(f"GeM Input Shape: {dummy_input.shape}")
    print(f"GeM Output Shape: {output.shape}")

    # GeM should reduce spatial dimensions to 1x1
    assert output.shape == (2, 64, 1, 1), f"GeM output shape mismatch: {output.shape}"

    print("Model components verification passed.")


def demo_training_pipeline():
    """Demonstrate the full training pipeline."""
    print("\n=== 3. Running Training Pipeline Demo ===")

    # 1. Setup Configuration for Speed
    CFG.debug = True  # Limits dataset size (Train: 100, Val: 50)
    CFG.epochs = 1
    CFG.working_dir = "./working/demo_run"
    CFG.output_dir = CFG.working_dir
    CFG.submission_path = os.path.join(CFG.working_dir, "submission.csv")

    if os.path.exists(CFG.working_dir):
        shutil.rmtree(CFG.working_dir)
    os.makedirs(CFG.working_dir, exist_ok=True)

    seed_everything(CFG.seed)

    # 2. Get DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_loaders()

    # Verify Data
    images, targets = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (
        CFG.batch_size,
        3,
        CFG.img_size,
        CFG.img_size,
    ), "Image batch shape mismatch"
    assert targets.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), "Target batch shape mismatch"

    # 3. Initialize Model
    # Using the first backbone from config
    backbone_name = CFG.backbones[0]
    print(f"Initializing model with backbone: {backbone_name}")
    model = AppleDiseaseModel(model_name=backbone_name, pretrained=True)
    model.to(CFG.device)

    # Verify Forward Pass
    with torch.no_grad():
        images = images.to(CFG.device)
        logits = model(images)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), "Model output shape mismatch"

    # 4. Run Training (fit_model)
    print("Starting training (1 epoch)...")
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)

    best_f1, save_path = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=CFG.device,
        epochs=CFG.epochs,
        model_name="demo_model",
        patience=1,
    )

    print(f"Training finished. Best F1: {best_f1}")
    print(f"Model saved to: {save_path}")

    assert os.path.exists(save_path), "Model checkpoint file was not created."

    return save_path, test_loader, backbone_name


def demo_inference_pipeline(model_path, test_loader, backbone_name):
    """Demonstrate the inference and submission generation pipeline."""
    print("\n=== 4. Running Inference Pipeline Demo ===")

    # Configuration for inference
    # We use the model trained in the previous step
    model_configs = [(backbone_name, model_path)]

    # Generate Submission
    print("Generating submission...")
    generate_submission(model_configs, test_loader, CFG.device)

    # Verify Submission File
    assert os.path.exists(CFG.submission_path), "Submission file not found."

    df_sub = pd.read_csv(CFG.submission_path)
    print("Submission file loaded successfully.")
    print(df_sub.head())

    # Check dimensions
    # In debug mode, test set is truncated to 50 samples
    expected_rows = 50 if CFG.debug else 3727
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
    assert list(df_sub.columns) == ["image", "labels"], "Submission columns mismatch"

    # Check label format (should be string, even if empty or single class)
    assert (
        df_sub["labels"].dtype == object
    ), "Labels column should be object/string type"

    print("Inference verification passed.")


if __name__ == "__main__":
    print("Starting Apple Disease Detection Library Demo...")

    # 1. Verify Utils
    demo_utils()

    # 2. Verify Model Components
    demo_model_components()

    # 3. Run Training Pipeline
    # Returns the path to the saved model and necessary objects for inference
    saved_model_path, test_loader_obj, backbone_name = demo_training_pipeline()

    # 4. Run Inference Pipeline
    demo_inference_pipeline(saved_model_path, test_loader_obj, backbone_name)

    print("\n=== All Demonstrations Completed Successfully ===")
