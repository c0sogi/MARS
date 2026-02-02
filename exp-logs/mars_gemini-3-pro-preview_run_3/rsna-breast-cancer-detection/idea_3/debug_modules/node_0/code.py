import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# 1. Import Config and Override for Speed/Demo
from library.config import Config

print(">>> Configuring environment for demonstration...")
# Override Config for rapid execution
Config.DEBUG = True
Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples
Config.EPOCHS = 1  # Run only 1 epoch
Config.BATCH_SIZE = 2  # Small batch size
Config.NUM_WORKERS = 0  # Main process only to avoid overhead
Config.PRETRAINED = False  # Skip downloading weights for speed
Config.PATIENCE = 1  # Early stopping check

# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# 2. Import Library Modules
from library.utils import seed_everything, probabilistic_f1
from library.preprocessing import process_image
from library.dataset import get_dataloaders
from library.model import BreastCancerModel
from library.train import Trainer
from library.inference import predict_and_submit


def run_demo():
    print("\n=== 1. Validating Utilities ===")
    seed_everything(Config.SEED)

    # Test Probabilistic F1 Score
    # Case 1: Perfect prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    pf1_perfect = probabilistic_f1(y_true, y_pred_perfect)
    assert np.isclose(pf1_perfect, 1.0), f"Expected pF1=1.0, got {pf1_perfect}"

    # Case 2: Worst prediction
    y_pred_worst = np.array([0.0, 1.0, 0.0, 1.0])
    pf1_worst = probabilistic_f1(y_true, y_pred_worst)
    assert np.isclose(pf1_worst, 0.0), f"Expected pF1=0.0, got {pf1_worst}"

    print("PASS: probabilistic_f1 logic verified.")

    print("\n=== 2. Validating Preprocessing ===")
    # Load metadata to find a valid image path
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    sample_path = os.path.join(Config.INPUT_DIR, df_train.iloc[0]["file_path"])

    print(f"Processing sample image: {sample_path}")
    tensor_img = process_image(sample_path)

    # Verify Tensor properties
    assert isinstance(tensor_img, torch.Tensor), "Output is not a torch.Tensor"
    assert tensor_img.shape == (
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Expected shape (3, {Config.IMG_HEIGHT}, {Config.IMG_WIDTH}), got {tensor_img.shape}"
    assert tensor_img.dtype == torch.float32, "Expected float32 dtype"

    print("PASS: Image processing pipeline verified.")

    print("\n=== 3. Validating Dataset & DataLoaders ===")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")
    assert images.shape[0] == Config.BATCH_SIZE
    assert images.shape[1] == 3
    assert labels.shape == (Config.BATCH_SIZE, 1)

    # Verify Test Loader (returns image, prediction_id)
    test_images, pred_ids = next(iter(test_loader))
    print(f"Test Batch - Images: {test_images.shape}, IDs: {len(pred_ids)}")
    assert len(pred_ids) == Config.BATCH_SIZE

    print("PASS: DataLoaders verified.")

    print("\n=== 4. Validating Model Architecture ===")
    model = BreastCancerModel(pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    with torch.no_grad():
        # Pass the batch retrieved earlier
        images = images.to(Config.DEVICE)
        logits = model(images)

    print(f"Model Output Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"

    print("PASS: Model architecture verified.")

    print("\n=== 5. Validating Training Loop ===")
    print("Initializing Trainer (Running 1 epoch on subset)...")
    trainer = Trainer()

    # Run training
    trainer.fit()

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created!"
    print(f"PASS: Training completed and checkpoint saved at {checkpoint_path}")

    print("\n=== 6. Validating Inference Pipeline ===")
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    predict_and_submit(
        checkpoint_path=checkpoint_path,
        output_path=submission_path,
        test_meta_path=Config.TEST_META_PATH,
        debug=True,
        subset_size=Config.DEBUG_SUBSET_SIZE,
    )

    assert os.path.exists(submission_path), "Submission file not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    assert "prediction_id" in df_sub.columns
    assert "cancer" in df_sub.columns
    assert len(df_sub) > 0

    print("PASS: Inference pipeline verified.")
    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
