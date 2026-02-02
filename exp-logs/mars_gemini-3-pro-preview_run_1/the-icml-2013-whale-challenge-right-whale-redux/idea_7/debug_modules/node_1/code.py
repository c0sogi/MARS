import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Import Config and Setup Overrides
from library.config import Config

# Define a separate working directory for this demo to avoid conflicts
DEMO_DIR = "./working/demo_run_script"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR)

# Override Config paths to use the demo directory
Config.WORKING_DIR = DEMO_DIR
Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# Create small metadata subsets to ensure data processing is fast (Optimize for Speed)
print("Creating small metadata subsets for demonstration...")
try:
    # Read original metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # Slice to create a tiny dataset
    small_train_path = os.path.join(DEMO_DIR, "train_small.csv")
    small_val_path = os.path.join(DEMO_DIR, "val_small.csv")
    small_test_path = os.path.join(DEMO_DIR, "test_small.csv")

    train_df.head(50).to_csv(small_train_path, index=False)
    val_df.head(20).to_csv(small_val_path, index=False)
    test_df.head(20).to_csv(small_test_path, index=False)

    # Point Config to these new small CSVs
    Config.TRAIN_CSV = small_train_path
    Config.VAL_CSV = small_val_path
    Config.TEST_CSV = small_test_path
    print("Metadata subsets created.")

except FileNotFoundError as e:
    print(f"Critical Error: Metadata files not found. {e}")
    exit(1)

# Override hyperparameters for the demo
Config.BATCH_SIZE = 8
Config.NUM_EPOCHS = 1
Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

# Import remaining library components
# Note: These imports must happen after Config modification if Config was a module,
# but since Config is a class, attributes are accessed at runtime.
from library.utils import set_seed, calculate_auc
from library.data_loader import get_dataloaders
from library.model import MultiResResNet34CRNN
from library.trainer import train_model, predict_and_submit


def run_demo():
    # Set reproducibility
    set_seed(Config.SEED)
    print(f"Demo initialized. Working directory: {Config.WORKING_DIR}")

    # ==========================================
    # 1. Verify Utilities
    # ==========================================
    print("\n--- 1. Verifying Utilities ---")
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2])
    auc = calculate_auc(y_true, y_pred)
    print(f"Calculated AUC: {auc}")
    assert 0.0 <= auc <= 1.0, "AUC calculation is out of bounds."
    assert auc > 0.5, "AUC should be high for this easy dummy case."

    # ==========================================
    # 2. Verify Data Loading & Processing
    # ==========================================
    print("\n--- 2. Verifying Data Loading ---")
    # load_cached_data=False forces processing of our new small CSVs
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug_subset=False
    )

    # Fetch a batch to verify shapes
    batch = next(iter(train_loader))
    inputs, targets_a, targets_b, lam = batch

    print(f"Input Batch Shape: {inputs.shape}")
    print(f"Targets A Shape: {targets_a.shape}")
    print(f"Mixup Lambda: {lam}")

    # Expected shape: (Batch, Channels, MelBins, Time)
    # Time ~ 2000Hz * 2s / 20 hop = 200 frames (+1 for centering usually) -> 201
    expected_shape = (Config.BATCH_SIZE, 3, 128, 201)
    assert (
        inputs.shape == expected_shape
    ), f"Input shape mismatch. Expected {expected_shape}, got {inputs.shape}"
    assert targets_a.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch."

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n--- 3. Verifying Model Architecture ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Instantiate model (pretrained=False for speed in demo)
    model = MultiResResNet34CRNN(pretrained=False).to(device)

    # Run dummy forward pass
    dummy_input = inputs.to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), "Model output should be (Batch, 1) logits."

    # ==========================================
    # 4. Verify Training Loop
    # ==========================================
    print("\n--- 4. Verifying Training Loop ---")
    # train_model handles the loop, saving, and validation
    # We use load_cached_data=True because get_dataloaders in step 2 already created the cache in DEMO_DIR
    model_path = train_model(debug=False, epochs=1, load_cached_data=True)

    assert os.path.exists(model_path), f"Model file was not saved at {model_path}"
    print(f"Training successful. Model saved to {model_path}")

    # ==========================================
    # 5. Verify Prediction & Submission
    # ==========================================
    print("\n--- 5. Verifying Prediction Pipeline ---")
    predict_and_submit(model_path, debug=False, load_cached_data=True)

    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Validate submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{sub_df.head()}")

    assert list(sub_df.columns) == [
        "clip",
        "probability",
    ], "Submission columns are incorrect."
    assert (
        len(sub_df) == 20
    ), f"Submission length mismatch. Expected 20 (from small_test.csv), got {len(sub_df)}"
    assert sub_df["probability"].dtype == float, "Probability column should be float."

    print("\nAll verification steps passed successfully.")


if __name__ == "__main__":
    run_demo()
