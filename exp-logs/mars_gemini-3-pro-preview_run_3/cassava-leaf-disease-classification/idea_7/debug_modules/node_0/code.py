import os
import sys
import pandas as pd
import torch
import warnings

# Ensure the current directory is in the path to locate the library modules
sys.path.append(".")

# Import necessary components from the provided library
from library.config import CFG, seed_everything
from library.data import get_loaders
from library.models import CassavaClassifier
from library.trainer import fit
from library.inference import generate_ensemble_submission

# Suppress warnings for a clean execution output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Cassava Leaf Disease Classification Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Define a working directory for this execution
    WORK_DIR = "./working/demo_execution"
    os.makedirs(WORK_DIR, exist_ok=True)

    # Override CFG settings for a fast, minimal execution
    CFG.debug = True  # Use subsampled data
    CFG.epochs = 1  # Train for only 1 epoch
    CFG.batch_size = 8  # Small batch size for speed and memory safety
    CFG.num_workers = 2  # Use available vCPUs
    CFG.output_dir = WORK_DIR
    CFG.submission_dir = WORK_DIR
    CFG.submission_file = os.path.join(WORK_DIR, "submission.csv")
    CFG.tta_steps = 1  # Disable TTA (1 view) for faster inference

    # -------------------------------------------------------------------------
    # 2. Data Consistency Setup
    # -------------------------------------------------------------------------
    # The inference library reads the test CSV directly from disk to build the submission.
    # In debug mode, get_loaders() subsamples the data. To ensure the loader and
    # the submission template match, we create a specific debug test CSV.
    print("[2] Preparing consistent test metadata...")
    original_test_csv = "./metadata/test.csv"

    if not os.path.exists(original_test_csv):
        raise FileNotFoundError(f"Metadata not found at {original_test_csv}")

    df_test = pd.read_csv(original_test_csv)
    # Sample 50 rows to match the logic in library.data for debug mode
    df_test_sample = df_test.sample(
        n=min(len(df_test), 50), random_state=CFG.seed
    ).reset_index(drop=True)

    debug_test_csv_path = os.path.join(WORK_DIR, "test_debug.csv")
    df_test_sample.to_csv(debug_test_csv_path, index=False)

    # Point CFG to this new consistent file
    CFG.test_csv = debug_test_csv_path
    print(
        f"    Created debug test CSV at {debug_test_csv_path} with {len(df_test_sample)} rows."
    )

    # Apply seed for reproducibility
    seed_everything(CFG.seed)

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_loaders()

    # Verify loaders are not empty
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Verify batch dimensions
    imgs, lbls = next(iter(train_loader))
    print(f"    Batch Shape: {imgs.shape}")
    assert imgs.shape == (
        CFG.batch_size,
        3,
        CFG.img_size,
        CFG.img_size,
    ), "Incorrect image tensor shape."
    assert lbls.shape == (CFG.batch_size,), "Incorrect label tensor shape."
    print("    Data loading verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration (Model B)
    # -------------------------------------------------------------------------
    # We train Model B (EfficientNet) for 1 epoch to demonstrate the fit() function.
    print(f"\n[4] Training Model B ({CFG.model_b_name}) for 1 epoch...")

    # fit() trains the model and saves the best checkpoint to CFG.output_dir
    best_acc_b = fit(CFG.model_b_name, "model_b", train_loader, val_loader)

    # Verify checkpoint existence
    model_b_path = os.path.join(CFG.output_dir, "model_b_best.pth")
    # Fallback to checkpoint.pth if best.pth wasn't copied (rare case if acc=0)
    if not os.path.exists(model_b_path):
        model_b_path = os.path.join(CFG.output_dir, "model_b_checkpoint.pth")

    assert os.path.exists(model_b_path), "Model B checkpoint was not generated."
    print(f"    Model B training finished. Best Val Acc: {best_acc_b:.4f}")

    # -------------------------------------------------------------------------
    # 5. Ensemble Mocking (Model A)
    # -------------------------------------------------------------------------
    # To demonstrate the ensemble inference without training a large ViT model,
    # we instantiate Model A with random weights and save it as a checkpoint.
    print(f"\n[5] Creating dummy checkpoint for Model A ({CFG.model_a_name})...")

    # Use pretrained=False for speed (no download) since we just need a valid state dict structure
    model_a = CassavaClassifier(CFG.model_a_name, CFG.num_classes, pretrained=False)

    state_a = {
        "state_dict": model_a.state_dict(),
        "best_acc": 0.0,
        "epoch": 0,
        "optimizer": {},
    }
    model_a_path = os.path.join(CFG.output_dir, "model_a_best.pth")
    torch.save(state_a, model_a_path)

    # Free memory
    del model_a
    torch.cuda.empty_cache()

    assert os.path.exists(model_a_path), "Model A dummy checkpoint creation failed."
    print("    Model A checkpoint ready.")

    # -------------------------------------------------------------------------
    # 6. Ensemble Inference
    # -------------------------------------------------------------------------
    print("\n[6] Running Ensemble Inference Pipeline...")

    # This function loads both models, runs inference, averages predictions, and saves submission
    generate_ensemble_submission(model_a_path, model_b_path, test_loader)

    # -------------------------------------------------------------------------
    # 7. Verification
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Submission File...")

    if not os.path.exists(CFG.submission_file):
        raise FileNotFoundError(f"Submission file not found at {CFG.submission_file}")

    df_sub = pd.read_csv(CFG.submission_file)
    print(f"    Submission Shape: {df_sub.shape}")
    print(f"    Columns: {list(df_sub.columns)}")

    # Verify row count matches the debug test set
    expected_rows = len(df_test_sample)
    assert (
        len(df_sub) == expected_rows
    ), f"Row count mismatch: Expected {expected_rows}, got {len(df_sub)}"

    # Verify label validity
    unique_labels = df_sub["label"].unique()
    assert all(
        l in range(5) for l in unique_labels
    ), "Submission contains invalid label codes."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
