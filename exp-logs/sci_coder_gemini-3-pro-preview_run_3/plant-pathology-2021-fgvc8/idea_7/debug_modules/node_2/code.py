import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_f1_score, MixupCutmix
from library.dataset import get_loaders, get_test_loader, AppleDataset
from library.model import AppleDiseaseModel
from library.engine import train_model, validate


def main():
    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    print("Setting up configuration for demonstration...")

    # Set seeds for reproducibility
    seed_everything(42)

    # Define temporary directories for this demo run
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    # Override Config paths to use demo directories
    Config.WORKING_DIR = DEMO_DIR
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.CACHE_DIR = DEMO_DIR  # Store parquet caches here

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Override Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2
    Config.IMAGE_SIZE = 256  # Smaller size for faster processing
    Config.MODEL_NAME = "resnet18"  # Smaller model for demo speed

    # ==========================================
    # 2. Data Preparation (Subsetting)
    # ==========================================
    print("Creating data subsets...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create subsets (20 train, 10 val, 10 test)
    # We ensure we pick existing files
    demo_train = orig_train.head(20).copy()
    demo_val = orig_val.head(10).copy()
    demo_test = orig_test.head(10).copy()

    # Save subsets
    demo_train_path = os.path.join(DEMO_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_DIR, "val.csv")
    demo_test_path = os.path.join(DEMO_DIR, "test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    # Point Config to these new files
    Config.TRAIN_METADATA = demo_train_path
    Config.VAL_METADATA = demo_val_path
    Config.TEST_METADATA = demo_test_path

    print(
        f"Train subset: {len(demo_train)}, Val subset: {len(demo_val)}, Test subset: {len(demo_test)}"
    )

    # ==========================================
    # 3. Component Verification: Dataset & Loader
    # ==========================================
    print("Verifying Dataset and Loaders...")

    # Initialize loaders (load_cached_data=False forces processing of our new subset CSVs)
    train_loader, val_loader = get_loaders(load_cached_data=False)

    # Fetch one batch
    images, targets = next(iter(train_loader))

    # Assertions
    print(f"Batch shapes - Images: {images.shape}, Targets: {targets.shape}")
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect Image Batch Shape"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect Target Batch Shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert targets.dtype == torch.float32, "Targets should be float32"

    # ==========================================
    # 4. Component Verification: Model
    # ==========================================
    print("Verifying Model...")

    device = Config.DEVICE
    model = AppleDiseaseModel(
        model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES, pretrained=False
    )
    model.to(device)

    # Dummy forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
        outputs = model(dummy_input)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"

    # ==========================================
    # 5. Component Verification: Utils
    # ==========================================
    print("Verifying Utils...")

    # Test MixupCutmix
    mixup_fn = MixupCutmix(prob=1.0, switch_prob=0.5)  # Force augmentation
    dummy_imgs = torch.randn(4, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    dummy_targets = torch.zeros(4, Config.NUM_CLASSES)
    dummy_targets[:, 0] = 1  # Set first class

    mixed_imgs, mixed_targets = mixup_fn(dummy_imgs, dummy_targets)

    assert mixed_imgs.shape == dummy_imgs.shape, "Mixup altered image shape"
    assert mixed_targets.shape == dummy_targets.shape, "Mixup altered target shape"

    # Test F1 Score
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array(
        [[0.9, 0.1, 0.8], [0.2, 0.7, 0.1]]
    )  # Should match perfectly with threshold 0.5
    f1 = calculate_f1_score(y_true, y_pred, threshold=0.5)
    assert f1 == 1.0, f"F1 calculation failed. Expected 1.0, got {f1}"

    # ==========================================
    # 6. Training Loop Execution
    # ==========================================
    print("Starting Training Loop Demonstration...")

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Run Training
    best_f1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=1,
    )

    print(f"Training finished. Best F1: {best_f1}")

    # Check if checkpoint was saved
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."

    # ==========================================
    # 7. Inference & Submission
    # ==========================================
    print("Generating Submission...")

    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Get Test Loader
    test_loader = get_test_loader(load_cached_data=False)

    # Inference Loop
    results = []

    # We manually iterate to get IDs (validate() doesn't return IDs)
    with torch.no_grad():
        for images, _, image_ids in test_loader:
            images = images.to(device)

            # Simple inference (no TTA for speed in demo, though Config.USE_TTA might be True)
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Convert to binary
            preds = (probs > Config.THRESHOLD).int().cpu().numpy()

            for i, pred in enumerate(preds):
                # Convert binary vector back to labels
                label_indices = np.where(pred == 1)[0]
                if len(label_indices) == 0:
                    label_str = "healthy"  # Fallback if no class predicted
                else:
                    label_list = [Config.LABELS[idx] for idx in label_indices]
                    label_str = " ".join(label_list)

                results.append({"image": image_ids[i], "labels": label_str})

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Save Submission
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")
    print("Top 5 rows of submission:")
    print(submission_df.head())

    assert len(submission_df) == len(demo_test), "Submission row count mismatch"
    assert os.path.exists(sub_path), "Submission file not created"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
