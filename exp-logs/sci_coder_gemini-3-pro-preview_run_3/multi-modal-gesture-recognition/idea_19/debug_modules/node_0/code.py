import os
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import Library Components
# Note: We import Config first to patch it before other modules use it
from library.config import Config
from library.dataset import GestureDataset
from library.model import ASK_RN
from library.loss import MultiTaskRefinementLoss
from library.trainer import Trainer
from library.inference import InferenceEngine
from library.data_utils import seed_everything


def setup_demo_environment():
    """
    Sets up a lightweight environment for the demo:
    1. Creates a working directory for demo outputs.
    2. Creates mini-versions of metadata CSVs to speed up data loading.
    3. Patches the Config class to use these mini-files and reduce training parameters.
    """
    print(">>> Setting up demo environment...")

    # Define paths
    demo_dir = "./working/demo_env"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    cache_dir = os.path.join(demo_dir, "cache")
    submission_dir = os.path.join(demo_dir, "submission")
    metadata_dir = os.path.join(demo_dir, "metadata")

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Create Mini Metadata (First 5 samples from each)
    # This prevents the data loader from processing hundreds of files
    for split in ["train", "val", "test"]:
        src_csv = os.path.join(Config.METADATA_DIR, f"{split}.csv")
        dst_csv = os.path.join(metadata_dir, f"{split}.csv")

        if os.path.exists(src_csv):
            df = pd.read_csv(src_csv)
            # Take top 5 samples
            mini_df = df.head(5)
            mini_df.to_csv(dst_csv, index=False)
            print(f"    Created mini {split}.csv with {len(mini_df)} samples.")
        else:
            print(f"    Warning: Source {split}.csv not found.")

    # Patch Config
    print("    Patching Config for speed...")
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = cache_dir
    Config.SUBMISSION_DIR = submission_dir
    Config.SUBMISSION_FILE = os.path.join(submission_dir, "submission.csv")

    Config.TRAIN_CSV = os.path.join(metadata_dir, "train.csv")
    Config.VAL_CSV = os.path.join(metadata_dir, "val.csv")
    Config.TEST_CSV = os.path.join(metadata_dir, "test.csv")

    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print(">>> Environment setup complete.\n")


def demonstrate_dataset():
    """
    Demonstrates loading the dataset and verifying item shapes.
    """
    print(">>> Demonstrating GestureDataset...")

    # Initialize dataset (uses the mini csv patched in Config)
    # augment=False for deterministic check
    ds = GestureDataset(split_name="train", augment=False, load_cached=False)

    print(f"    Dataset length (windows): {len(ds)}")
    if len(ds) == 0:
        raise ValueError("Dataset is empty. Check mini-metadata creation.")

    # Fetch one item
    item = ds[0]

    # Verify keys
    expected_keys = {
        "features",
        "cls_labels",
        "bnd_labels",
        "sample_id",
        "window_start",
    }
    assert expected_keys.issubset(
        item.keys()
    ), f"Missing keys in dataset item. Found: {item.keys()}"

    # Verify shapes
    # Features: (Window_Size, Input_Dim) -> (64, 193)
    features = item["features"]
    cls_labels = item["cls_labels"]
    bnd_labels = item["bnd_labels"]

    print(f"    Features Shape: {features.shape}")
    print(f"    Class Labels Shape: {cls_labels.shape}")
    print(f"    Boundary Labels Shape: {bnd_labels.shape}")

    assert features.shape == (
        Config.WINDOW_SIZE,
        Config.TOTAL_INPUT_DIM,
    ), f"Mismatch in feature shape. Expected {(Config.WINDOW_SIZE, Config.TOTAL_INPUT_DIM)}, got {features.shape}"
    assert cls_labels.shape == (Config.WINDOW_SIZE,), "Mismatch in cls_labels shape"
    assert bnd_labels.shape == (Config.WINDOW_SIZE,), "Mismatch in bnd_labels shape"

    print(">>> Dataset demonstration successful.\n")


def demonstrate_model_and_loss():
    """
    Demonstrates Model instantiation, Forward pass, and Loss calculation.
    """
    print(">>> Demonstrating ASK_RN Model and Loss...")

    device = Config.DEVICE
    model = ASK_RN().to(device)
    criterion = MultiTaskRefinementLoss().to(device)

    # Create Dummy Batch
    batch_size = 2
    dummy_features = torch.randn(
        batch_size, Config.WINDOW_SIZE, Config.TOTAL_INPUT_DIM
    ).to(device)

    # Forward Pass
    outputs = model(dummy_features)

    # Verify Output Keys
    expected_outputs = {"logits_s1", "logits_bnd", "logits_s2", "logits_s3"}
    assert expected_outputs.issubset(
        outputs.keys()
    ), "Model output missing required keys."

    # Verify Output Shapes
    # Logits should be (Batch, Frames, Num_Classes) or (Batch, Frames, 1)
    s3_shape = outputs["logits_s3"].shape
    print(f"    Stage 3 Logits Shape: {s3_shape}")
    assert s3_shape == (batch_size, Config.WINDOW_SIZE, Config.NUM_CLASSES)

    # Calculate Loss
    # Create dummy targets
    dummy_cls = (
        torch.randint(0, Config.NUM_CLASSES, (batch_size, Config.WINDOW_SIZE))
        .long()
        .to(device)
    )
    dummy_bnd = torch.randint(0, 2, (batch_size, Config.WINDOW_SIZE)).float().to(device)

    targets = {"cls_labels": dummy_cls, "bnd_labels": dummy_bnd}

    loss, metrics = criterion(outputs, targets)

    print(f"    Calculated Loss: {loss.item():.4f}")
    print(f"    Metrics: {metrics}")

    assert not torch.isnan(loss), "Loss is NaN"

    print(">>> Model and Loss demonstration successful.\n")


def demonstrate_training():
    """
    Demonstrates the Trainer class.
    Runs a short training loop using the mini-dataset.
    """
    print(">>> Demonstrating Trainer...")

    # Initialize Trainer
    # load_cached_data=False forces processing of our new mini-csvs
    trainer = Trainer(load_cached_data=False)

    # Run Fit
    # Config.NUM_EPOCHS is set to 1 in setup
    trainer.fit()

    # Check if model was saved
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if os.path.exists(model_path):
        print(f"    Model successfully saved to {model_path}")
    else:
        raise FileNotFoundError("Trainer failed to save the best model.")

    print(">>> Training demonstration successful.\n")


def demonstrate_inference():
    """
    Demonstrates the InferenceEngine.
    Loads the model trained in the previous step and generates predictions.
    """
    print(">>> Demonstrating InferenceEngine...")

    # Initialize Engine
    # This automatically loads the 'best_model.pth' from Config.CACHE_DIR
    engine = InferenceEngine()

    # Run Inference on Mini Test Set
    engine.run_inference(load_cached_data=False)

    # Verify Submission File
    sub_file = Config.SUBMISSION_FILE
    if os.path.exists(sub_file):
        with open(sub_file, "r") as f:
            lines = f.readlines()
        print(f"    Submission file created with {len(lines)} lines.")

        if len(lines) > 0:
            print(f"    Sample Output: {lines[0].strip()}")

            # Basic format check: SessionID,Labels
            parts = lines[0].strip().split(",")
            assert len(parts) >= 1, "Invalid submission format"
            assert (
                "Sample" in parts[0] or "Session" in parts[0]
            ), "Invalid Sequence ID format"
    else:
        raise FileNotFoundError("InferenceEngine failed to create submission file.")

    # Demonstrate explicit sequence prediction logic
    print("    Verifying explicit prediction logic...")
    dummy_input = np.random.randn(100, Config.TOTAL_INPUT_DIM).astype(np.float32)
    pred_labels = engine.predict_sequence(dummy_input)
    decoded = engine.decode_predictions(pred_labels)

    print(f"    Dummy Input (T=100) -> Predicted Labels Shape: {pred_labels.shape}")
    print(f"    Decoded Gestures: {decoded}")

    assert pred_labels.shape == (100,), "Prediction shape mismatch"
    assert isinstance(decoded, list), "Decoding should return a list"

    print(">>> Inference demonstration successful.\n")


if __name__ == "__main__":
    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Dataset
        demonstrate_dataset()

        # 3. Model
        demonstrate_model_and_loss()

        # 4. Training
        demonstrate_training()

        # 5. Inference
        demonstrate_inference()

        print("=== All Demonstrations Completed Successfully ===")

    except Exception as e:
        print(f"\n!!! Demo Failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
