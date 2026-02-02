import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
from torch.optim import AdamW

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, get_class_weights
from library.data import (
    load_dataset_dataframe,
    get_loaders,
    AppleDataset,
    get_transforms,
)
from library.model import AppleDiseaseFPN
from library.loss import DeepSupervisionLoss
from library.train import train_one_epoch, validate
from library.inference import predict_with_tta


def run_demo():
    # 1. Setup and Configuration Overrides
    print(">>> Setting up demonstration configuration...")
    seed_everything(Config.seed)

    # Create a demo working directory
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config for speed
    Config.working_dir = demo_dir
    Config.cache_dir = demo_dir
    Config.epochs = 1
    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the first model in the list for demonstration
    model_cfg = Config.models[0]
    # Reduce batch size for demo to ensure it runs even on smaller VRAM if needed,
    # though A100 is available.
    model_cfg["batch_size"] = 4

    logger = get_logger("Demo")
    logger.info(f"Running on device: {Config.device}")

    # 2. Data Loading
    print("\n>>> Loading and subsetting data...")
    # Load real metadata
    df_train_full = load_dataset_dataframe("train", load_cached_data=False)
    df_val_full = load_dataset_dataframe("val", load_cached_data=False)

    # Subset data for speed (e.g., 20 samples)
    df_train_demo = df_train_full.head(20).reset_index(drop=True)
    df_val_demo = df_val_full.head(10).reset_index(drop=True)

    # Create a dummy test dataframe based on val structure for inference demo
    df_test_demo = df_val_demo.copy()
    df_test_demo = df_test_demo[["image_id", "file_path"]]  # Test only has these cols

    logger.info(f"Train subset size: {len(df_train_demo)}")
    logger.info(f"Val subset size: {len(df_val_demo)}")

    # 3. DataLoader Instantiation
    print("\n>>> Creating DataLoaders...")
    train_loader, val_loader, test_loader = get_loaders(
        df_train_demo, df_val_demo, df_test_demo, model_cfg
    )

    # Verify DataLoader
    images, labels = next(iter(train_loader))
    logger.info(f"Batch Image Shape: {images.shape}")
    logger.info(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (
        model_cfg["batch_size"],
        3,
        model_cfg["img_size"],
        model_cfg["img_size"],
    ), "Incorrect image batch shape"
    assert labels.shape == (
        model_cfg["batch_size"],
        Config.num_classes,
    ), "Incorrect label batch shape"

    # 4. Model Instantiation and Verification
    print("\n>>> Instantiating Model...")
    # Using pretrained=False to avoid downloading weights during this short demo execution
    model = AppleDiseaseFPN(
        model_name=model_cfg["name"], num_classes=Config.num_classes, pretrained=False
    )
    model.to(Config.device)

    # Verify Forward Pass (Training Mode - Deep Supervision)
    model.train()
    images = images.to(Config.device)
    outputs = model(images)

    # Should return a tuple of 3 tensors (p3, p4, p5)
    assert isinstance(
        outputs, tuple
    ), "Model in train mode should return a tuple (Deep Supervision)"
    assert len(outputs) == 3, f"Expected 3 outputs from FPN, got {len(outputs)}"
    logger.info("Model training forward pass successful (Tuple output verified).")

    # Verify Forward Pass (Eval Mode)
    model.eval()
    with torch.no_grad():
        output_eval = model(images)
    assert isinstance(
        output_eval, torch.Tensor
    ), "Model in eval mode should return a Tensor"
    assert output_eval.shape == (
        model_cfg["batch_size"],
        Config.num_classes,
    ), "Model eval output shape mismatch"
    logger.info("Model eval forward pass successful.")

    # 5. Loss Function Verification
    print("\n>>> Verifying Loss Function...")
    # Compute class weights (mocking logic)
    class_weights = get_class_weights(df_train_demo, load_cached_data=False)
    criterion = DeepSupervisionLoss(class_weights=class_weights)

    # Calculate loss using the training outputs (tuple)
    labels = labels.to(Config.device)
    loss = criterion(outputs, labels)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    logger.info(f"Computed Loss: {loss.item():.4f}")

    # 6. Training Loop Simulation
    print("\n>>> Simulating Training Epoch...")
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_amp)

    # Train one epoch
    avg_train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, Config.device, scaler
    )
    logger.info(f"Average Train Loss: {avg_train_loss:.4f}")

    # Validate
    print("\n>>> Simulating Validation...")
    val_loss, val_auc = validate(model, val_loader, criterion, Config.device)
    logger.info(f"Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}")

    # 7. Inference Simulation
    print("\n>>> Simulating Inference with TTA...")
    # predict_with_tta expects model, loader, device
    probs, ids = predict_with_tta(model, test_loader, Config.device)

    assert len(probs) == len(
        df_test_demo
    ), "Number of predictions does not match test set size"
    assert probs.shape[1] == Config.num_classes, "Prediction classes dimension mismatch"
    assert len(ids) == len(df_test_demo), "Number of IDs does not match test set size"

    logger.info(f"Inference generated {len(probs)} predictions.")
    logger.info(f"Sample prediction for {ids[0]}: {probs[0]}")

    # 8. Generate Submission File (Demo)
    print("\n>>> Generating Demo Submission...")
    submission_rows = []
    for img_id, prob_vec in zip(ids, probs):
        row = {"image_id": img_id}
        for idx, label in enumerate(Config.class_labels):
            row[label] = prob_vec[idx]
        submission_rows.append(row)

    sub_df = pd.DataFrame(submission_rows)
    sub_path = os.path.join(Config.working_dir, "demo_submission.csv")
    sub_df.to_csv(sub_path, index=False)
    logger.info(f"Demo submission saved to {sub_path}")

    print("\n>>> Demonstration Complete. All checks passed.")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\n!!! Demo Failed with Error: {e}")
        # Re-raise to ensure the task fails if the code is broken
        raise e
