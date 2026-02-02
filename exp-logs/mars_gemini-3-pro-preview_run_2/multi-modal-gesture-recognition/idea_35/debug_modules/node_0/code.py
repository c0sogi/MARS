import os
import torch
import shutil
import warnings
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import get_dataloaders
from library.model import CRGCN
from library.losses import compute_total_loss
from library.trainer import Trainer
from library.inference import generate_submission

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment...")

    # Set a separate working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    # Override Config parameters
    Config.WORKING_DIR = DEMO_DIR
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce workload for demonstration
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.DEBUG_SAMPLE_SIZE = 4  # Only use 4 samples
    Config.LSTM_HIDDEN_DIM = 64  # Smaller model for speed
    Config.TCN_CHANNELS = 64
    Config.TCN_NUM_LAYERS = 4

    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Batch Size: {Config.BATCH_SIZE}, Epochs: {Config.NUM_EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Data Loader Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Loaders...")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        debug_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Fetch one batch
    try:
        batch = next(iter(train_loader))
        features, labels, boundaries, mask, lengths = batch
    except StopIteration:
        raise RuntimeError("Data loader returned no data. Check input files.")

    # Validate shapes
    B, T, D = features.shape
    print(
        f"Batch Shapes -> Features: {features.shape}, Labels: {labels.shape}, Mask: {mask.shape}"
    )

    assert B <= Config.BATCH_SIZE, f"Batch size {B} exceeds config {Config.BATCH_SIZE}"
    assert (
        D == Config.INPUT_DIM
    ), f"Input dimension {D} does not match config {Config.INPUT_DIM}"
    assert labels.shape == (B, T), "Labels shape mismatch"
    assert boundaries.shape == (B, T), "Boundaries shape mismatch"
    assert mask.shape == (B, T), "Mask shape mismatch"

    print("Data Loader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CRGCN().to(device)

    # Move batch to device
    features = features.to(device)
    labels = labels.to(device)
    boundaries = boundaries.to(device)
    mask = mask.to(device)

    # Forward pass
    outputs = model(features, mask)

    # Validate outputs
    assert isinstance(outputs, dict), "Model output should be a dictionary"
    assert (
        "stage1" in outputs and "stage2" in outputs and "stage3" in outputs
    ), "Missing stages in output"

    # Check Stage 3 logits
    s3_cls, s3_bnd = outputs["stage3"]
    # Logits shape: (B, NumClasses, T) and (B, 1, T)
    assert s3_cls.shape == (
        B,
        Config.NUM_CLASSES,
        T,
    ), f"Stage 3 CLS shape mismatch: {s3_cls.shape}"
    assert s3_bnd.shape == (B, 1, T), f"Stage 3 BND shape mismatch: {s3_bnd.shape}"

    print("Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 4. Loss Calculation
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Loss Computation...")

    loss, metrics = compute_total_loss(outputs, labels, boundaries, mask)

    print(f"Total Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    assert torch.is_tensor(loss), "Loss must be a tensor"
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("Loss computation verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logger = setup_logger(
        "demo_trainer", log_file=os.path.join(Config.WORKING_DIR, "train.log")
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        logger=logger,
    )

    # Run training
    trainer.fit()

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    print("Training loop completed successfully.")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running Inference...")

    # Run inference using the checkpoint generated above
    # Note: generate_submission re-loads the model and data internally
    generate_submission(debug_size=Config.DEBUG_SAMPLE_SIZE)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Read and verify submission content
    df_sub = pd.read_csv(submission_path, header=None)
    print(f"Submission file generated with {len(df_sub)} rows.")
    print("First 2 rows of submission:")
    print(df_sub.head(2))

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
