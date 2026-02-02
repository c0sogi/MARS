import os
import sys
import shutil
import pandas as pd
import torch
import torch.optim as optim
import numpy as np

# Import provided library modules
from library.config import Config
from library.dataset import get_dataloaders
from library.model import EfficientNetB4Unet
from library.loss import HybridLoss
from library.engine import train_one_epoch, evaluate
from library.inference import Predictor
from library.utils import seed_everything


def run_demo():
    print("Starting SIIM-FISABIO-RSNA COVID-19 Detection Pipeline Demo...")

    # 1. Setup & Configuration Overrides
    # We override Config to use a tiny subset of data and run fast
    seed_everything(Config.SEED)

    # Create working directories for demo
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    print("\n[1] Preparing Mini-Datasets for Speed...")

    # Load original metadata
    orig_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    orig_val = pd.read_csv(Config.VAL_METADATA_PATH)
    orig_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Take a tiny subset (e.g., 4 samples each) to ensure < 1 min runtime
    mini_train = orig_train.head(4).copy()
    mini_val = orig_val.head(4).copy()
    mini_test = orig_test.head(4).copy()

    # Save mini metadata
    mini_train_path = os.path.join(demo_dir, "train.csv")
    mini_val_path = os.path.join(demo_dir, "val.csv")
    mini_test_path = os.path.join(demo_dir, "test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Override Config paths to point to mini datasets and demo cache
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    Config.CACHE_TRAIN_IMAGES = os.path.join(demo_dir, "train_images.npy")
    Config.CACHE_TRAIN_MASKS = os.path.join(demo_dir, "train_masks.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(demo_dir, "train_labels.npy")

    Config.CACHE_VAL_IMAGES = os.path.join(demo_dir, "val_images.npy")
    Config.CACHE_VAL_MASKS = os.path.join(demo_dir, "val_masks.npy")
    Config.CACHE_VAL_LABELS = os.path.join(demo_dir, "val_labels.npy")

    Config.CACHE_TEST_IMAGES = os.path.join(demo_dir, "test_images.npy")
    Config.CACHE_TEST_DIMS = os.path.join(demo_dir, "test_dims.parquet")

    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Override hyperparameters
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo
    Config.BACKBONE = "efficientnet_b4"  # Fix model name format (Cite debug_lesson_1)

    print("    Config updated to use mini-datasets and demo paths.")

    # 2. Data Loading
    print("\n[2] Testing Data Loading (dataset.py)...")
    # load_cached_data=False forces processing from the CSVs we just created
    train_loader, val_loader = get_dataloaders(load_cached_data=False)

    # Verification
    assert len(train_loader) > 0, "Train loader is empty"
    images, masks, labels = next(iter(train_loader))

    print(
        f"    Batch Shapes -> Images: {images.shape}, Masks: {masks.shape}, Labels: {labels.shape}"
    )

    # Check dimensions: (B, 3, 512, 512), (B, 1, 512, 512), (B, 4)
    assert images.shape == (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert masks.shape == (Config.BATCH_SIZE, 1, Config.IMG_SIZE, Config.IMG_SIZE)
    assert labels.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)
    print("    Data Loading verification passed.")

    # 3. Model Initialization & Forward Pass
    print("\n[3] Testing Model Architecture (model.py)...")
    device = Config.DEVICE
    model = EfficientNetB4Unet().to(device)

    # Pass the batch from step 2
    images = images.to(device)

    # Model returns study_logits and a list of mask_logits (if deep supervision is on)
    model.train()  # Ensure training mode for deep supervision
    study_logits, mask_logits_list = model(images)

    print(f"    Study Logits Shape: {study_logits.shape}")
    print(f"    Mask Logits List Length: {len(mask_logits_list)}")
    print(f"    Primary Mask Logits Shape: {mask_logits_list[0].shape}")

    assert study_logits.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)
    assert isinstance(mask_logits_list, list)
    # With deep supervision, we expect 3 outputs (stride 1, 2, 4)
    assert len(mask_logits_list) == 3
    assert mask_logits_list[0].shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    print("    Model architecture verification passed.")

    # 4. Loss Function
    print("\n[4] Testing Loss Function (loss.py)...")
    criterion = HybridLoss()
    masks = masks.to(device)
    labels = labels.to(device)

    loss_dict = criterion(study_logits, mask_logits_list, labels, masks)

    print(f"    Total Loss: {loss_dict['loss'].item():.4f}")
    print(f"    Study Loss: {loss_dict['study_loss'].item():.4f}")
    print(f"    Seg Loss:   {loss_dict['seg_loss'].item():.4f}")

    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN"
    print("    Loss function verification passed.")

    # 5. Training Loop
    print("\n[5] Testing Training Loop (engine.py)...")
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run 1 epoch
    avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch=1)
    print(f"    Epoch 1 completed. Avg Loss: {avg_loss:.4f}")

    # Save the model for inference step
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"    Model saved to {Config.BEST_MODEL_PATH}")

    # 6. Evaluation
    print("\n[6] Testing Evaluation (engine.py)...")
    val_loss, val_map = evaluate(model, val_loader, device)
    print(f"    Validation complete. Loss: {val_loss:.4f}, mAP: {val_map:.4f}")

    # 7. Inference
    print("\n[7] Testing Inference Pipeline (inference.py)...")
    # Initialize Predictor (loads the model we just saved)
    predictor = Predictor(model_path=Config.BEST_MODEL_PATH)

    # Run submission generation
    predictor.generate_submission()

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission file generated at {Config.SUBMISSION_PATH}")
        print(f"    Rows: {len(sub_df)}")
        print(f"    Columns: {list(sub_df.columns)}")

        # We expect 2 rows per image in test set (1 study row + 1 image row)
        # We used 4 test images -> 8 rows total
        expected_rows = len(mini_test) * 2
        assert (
            len(sub_df) == expected_rows
        ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"
        assert "PredictionString" in sub_df.columns
        print("    Inference verification passed.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\nAll pipeline components verified successfully.")


if __name__ == "__main__":
    run_demo()
