import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, AverageMeter, Mixup
from library.dataset import get_dataloader
from library.model import PlantConvNeXt
from library.train import run_training
from library.inference import generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("==== Starting Plant Classification Demo ====")

    # ---------------------------------------------------------
    # 1. Configure for Fast Demonstration
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Override Config defaults to ensure speed and isolation
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2  # Reduce workers for small demo batch
    Config.PRETRAINED = False  # Skip downloading weights for speed/offline capability
    Config.WORK_DIR = "./working/demo_run"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORK_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.WORK_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=1, PRETRAINED=False")

    # ---------------------------------------------------------
    # 2. Verify Utilities
    # ---------------------------------------------------------
    print("\n[2] Verifying Utilities (AverageMeter, Mixup)...")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(val=10, n=2)  # sum=20, count=2
    meter.update(val=20, n=1)  # sum=40, count=3
    assert (
        abs(meter.avg - (40 / 3)) < 1e-6
    ), f"AverageMeter failed: expected {40/3}, got {meter.avg}"
    print("AverageMeter logic verified.")

    # Test Mixup
    mixup_fn = Mixup(prob=1.0, switch_prob=0.5, num_classes=10)  # Force augmentation
    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_targets = torch.tensor([0, 1, 2, 3])

    mixed_imgs, mixed_targets = mixup_fn(dummy_images, dummy_targets)

    assert mixed_imgs.shape == dummy_images.shape, "Mixup image shape mismatch"
    assert mixed_targets.shape == (
        4,
        10,
    ), "Mixup target should be one-hot encoded (N, num_classes)"
    print("Mixup transformation verified.")

    # ---------------------------------------------------------
    # 3. Verify Data Loading
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Loading...")

    train_loader = get_dataloader("train", debug=True, batch_size=Config.BATCH_SIZE)
    val_loader = get_dataloader("val", debug=True, batch_size=Config.BATCH_SIZE)

    # Fetch one batch
    imgs, lbls = next(iter(train_loader))

    print(f"Batch shapes - Images: {imgs.shape}, Labels: {lbls.shape}")

    # Assertions
    assert imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Incorrect image shape: {imgs.shape}"
    assert lbls.shape == (Config.BATCH_SIZE,), f"Incorrect label shape: {lbls.shape}"
    assert imgs.dtype == torch.float32, "Images should be float32"
    assert lbls.dtype == torch.long, "Labels should be long (int64)"

    print("DataLoader functional verification passed.")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = PlantConvNeXt(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.eval()

    # Run forward pass on the batch fetched earlier
    with torch.no_grad():
        logits = model(imgs)

    print(f"Logits shape: {logits.shape}")

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"

    print("Model forward pass verified.")

    # ---------------------------------------------------------
    # 5. Run Training Integration Test
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch, Debug Mode)...")

    # This runs the full training pipeline defined in library.train
    run_training(
        debug=Config.DEBUG, epochs=Config.EPOCHS, save_path=Config.BEST_MODEL_PATH
    )

    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Training failed to save the best model checkpoint."
    print(f"Training completed. Model saved to {Config.BEST_MODEL_PATH}")

    # ---------------------------------------------------------
    # 6. Run Inference Integration Test
    # ---------------------------------------------------------
    print("\n[6] Running Inference...")

    generate_submission(
        model_path=Config.BEST_MODEL_PATH,
        output_path=Config.SUBMISSION_FILE,
        debug=Config.DEBUG,
    )

    assert os.path.exists(
        Config.SUBMISSION_FILE
    ), "Inference failed to save submission file."

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    assert list(df_sub.columns) == ["Id", "Predicted"], "Submission columns mismatch"
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} predictions in debug mode, got {len(df_sub)}"
    assert not df_sub.isnull().values.any(), "Submission contains NaN values"

    print("Inference verification passed.")
    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
