import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, weighted_auc_score
from library.dataset import load_metadata, StegoDataset, get_transforms
from library.model import HPF_EfficientNet
from library.engine import train_one_epoch, valid_one_epoch, fit, predict


def main():
    print("=== Starting Steganography Detection Pipeline Demonstration ===")

    # ------------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------------
    # Override Config defaults for a fast demonstration (Speed Optimization)
    Config.debug = True
    Config.epochs = 1
    Config.batch_size = 8
    Config.train_subset_size = 32  # Small subset for demo
    Config.val_subset_size = 16
    Config.pretrained = False  # Disable download for speed/offline safety
    Config.num_workers = 2  # Reduce overhead for small batches

    # Initialize directories and seeds
    Config.setup()
    seed_everything(Config.seed)

    device = Config.device
    print(f"Configuration loaded. Device: {device}")
    print(f"Debug Mode: {Config.debug}, Batch Size: {Config.batch_size}")

    # ------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # ------------------------------------------------------------------------
    print("\n--- Verifying Data Loading ---")

    # Load metadata (debug mode triggers subsetting)
    df_train = load_metadata(
        Config.train_csv, debug=Config.debug, subset_size=Config.train_subset_size
    )
    df_val = load_metadata(
        Config.val_csv, debug=Config.debug, subset_size=Config.val_subset_size
    )

    # Verify metadata structure
    assert not df_train.empty, "Training dataframe is empty."
    assert (
        "file_path" in df_train.columns and "label" in df_train.columns
    ), "Missing required columns in train metadata."

    # Instantiate Datasets
    train_dataset = StegoDataset(df_train, transform=get_transforms(mode="train"))
    val_dataset = StegoDataset(df_val, transform=get_transforms(mode="val"))

    # Instantiate DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    # Verify Batch Structure
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (
        Config.batch_size,
        3,
        Config.img_size,
        Config.img_size,
    ), f"Expected image shape {(Config.batch_size, 3, Config.img_size, Config.img_size)}, got {images.shape}"
    assert labels.shape == (
        Config.batch_size,
    ), f"Expected label shape {(Config.batch_size,)}, got {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32 tensor."

    # ------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass Verification
    # ------------------------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")

    model = HPF_EfficientNet(pretrained=Config.pretrained).to(device)

    # Test forward pass with the loaded batch
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (
        Config.batch_size,
        Config.num_classes,
    ), f"Expected output shape {(Config.batch_size, Config.num_classes)}, got {outputs.shape}"

    # ------------------------------------------------------------------------
    # 4. Metric Verification
    # ------------------------------------------------------------------------
    print("\n--- Verifying Weighted AUC Metric ---")

    # Create synthetic data for metric testing
    # Case: Perfect prediction
    y_true_perfect = np.array([0, 0, 1, 1])
    y_score_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    score_perfect = weighted_auc_score(y_true_perfect, y_score_perfect)

    # Case: Random/Bad prediction
    y_true_bad = np.array([0, 0, 1, 1])
    y_score_bad = np.array([0.9, 0.8, 0.1, 0.2])
    score_bad = weighted_auc_score(y_true_bad, y_score_bad)

    print(f"Perfect Score: {score_perfect}")
    print(f"Bad Score: {score_bad}")

    assert np.isclose(score_perfect, 1.0), "Perfect predictions should yield AUC 1.0"
    assert isinstance(score_perfect, float), "Metric should return a float."

    # ------------------------------------------------------------------------
    # 5. Training Loop Demonstration (Engine)
    # ------------------------------------------------------------------------
    print("\n--- Running Training Loop (1 Epoch) ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Scheduler setup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.max_lr,
        epochs=Config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # Run 'fit' function from engine
    save_path = os.path.join(Config.working_dir, "best_model_demo.pth")

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.epochs,
        patience=Config.patience,
        save_path=save_path,
    )

    assert os.path.exists(save_path), "Model checkpoint was not saved."
    print("Training complete. Checkpoint saved.")

    # ------------------------------------------------------------------------
    # 6. Inference Demonstration
    # ------------------------------------------------------------------------
    print("\n--- Running Inference on Test Set ---")

    # Note: predict() internally loads test metadata and uses Config.submission_path
    predict(model_path=save_path, device=device, debug=True)

    assert os.path.exists(Config.submission_path), "Submission file was not created."

    # Verify submission format
    sub_df = pd.read_csv(Config.submission_path)
    print(f"Submission Head:\n{sub_df.head()}")

    assert list(sub_df.columns) == ["Id", "Label"], "Submission columns mismatch."
    assert len(sub_df) > 0, "Submission file is empty."
    assert sub_df["Label"].dtype == float, "Label column should be float."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
