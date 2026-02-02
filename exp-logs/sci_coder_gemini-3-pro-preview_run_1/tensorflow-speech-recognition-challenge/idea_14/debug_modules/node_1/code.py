import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.dataset import get_dataloaders
from library.model import DilatedEfficientNet
from library.trainer import Trainer
from library.utils import set_seed, DistillationLoss


def run_demo():
    print("============================================================")
    print("   Speech Command Recognition: Library Demo & Verification")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Demonstration
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo run...")

    # Modify Config class attributes directly to affect library modules
    Config.debug = True
    Config.debug_sample_size = 64  # Small subset for speed
    Config.batch_size = 8  # Small batch for demo
    Config.num_workers = 2

    # Reduce training duration
    Config.epochs_teacher = 1
    Config.epochs_student = 1

    # Set output directories to a demo folder
    Config.working_dir = "./working/demo_execution"
    Config.checkpoint_dir = Config.working_dir
    Config.submission_dir = Config.working_dir
    Config.submission_path = os.path.join(Config.submission_dir, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    print(f"    Working Directory: {Config.working_dir}")
    print(f"    Debug Mode: {Config.debug}")
    print(f"    Batch Size: {Config.batch_size}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[2] Initializing DataLoaders...")

    # This will load metadata, balance the training set, and return loaders
    train_loader, val_loader, test_loader, label_encoder = get_dataloaders(
        load_cached_data=False
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")
    print(f"    Classes:       {len(label_encoder.classes_)}")

    # Verify Batch Shapes
    # Expected Input: (Batch, 1, n_mels, time_steps)
    # n_mels = 128 (Config), time_steps approx 101 for 1s audio with hop 160
    inputs, targets = next(iter(train_loader))

    print(f"    Sample Input Shape: {inputs.shape}")
    print(f"    Sample Target Shape: {targets.shape}")

    assert inputs.dim() == 4, "Input must be 4D tensor (B, C, F, T)"
    assert inputs.size(1) == 1, "Input channel must be 1 (Mono MelSpec)"
    assert inputs.size(2) == Config.n_mels, f"Freq dim must be {Config.n_mels}"
    assert targets.size(0) == Config.batch_size, "Target batch size mismatch"

    print("    -> Data Loading Verified.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = Config.device
    model = DilatedEfficientNet(config=Config).to(device)

    # Create dummy input
    dummy_input = torch.randn(Config.batch_size, 1, Config.n_mels, 101).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"    Logits Shape: {logits.shape}")

    assert logits.size(0) == Config.batch_size
    assert logits.size(1) == Config.num_classes

    print("    -> Model Forward Pass Verified.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Distillation Loss...")

    distill_criterion = DistillationLoss(distillation_weight=0.5, temperature=2.0)

    # Dummy logits
    student_logits = torch.randn(Config.batch_size, Config.num_classes)
    teacher_logits = torch.randn(Config.batch_size, Config.num_classes)
    dummy_targets = torch.randint(0, Config.num_classes, (Config.batch_size,))

    # Calculate loss
    loss = distill_criterion(student_logits, teacher_logits, dummy_targets)

    print(f"    Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("    -> Loss Function Verified.")

    # -------------------------------------------------------------------------
    # 5. Full Pipeline Execution (Trainer)
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Pipeline (Teacher -> Student)...")

    trainer = Trainer(
        train_loader, val_loader, test_loader, label_encoder, config=Config
    )

    # Run the full pipeline:
    # 1. Train Teacher (1 epoch)
    # 2. Save Teacher
    # 3. Train Student (1 epoch) using Teacher
    # 4. Generate Submission
    trainer.run()

    print("    -> Pipeline Execution Completed.")

    # -------------------------------------------------------------------------
    # 6. Submission Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Submission Output...")

    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    df_sub = pd.read_csv(Config.submission_path)
    print(f"    Submission Rows: {len(df_sub)}")
    print(f"    Columns: {list(df_sub.columns)}")
    print("    Sample Rows:")
    print(df_sub.head())

    # Basic Checks
    assert "fname" in df_sub.columns
    assert "label" in df_sub.columns
    assert len(df_sub) > 0
    # Check if labels are within valid set (target labels + unknown + silence)
    valid_labels = set(
        Config.target_labels + [Config.silence_label, Config.unknown_label]
    )
    assert (
        df_sub["label"].isin(valid_labels).all()
    ), "Submission contains invalid labels"

    print("    -> Submission Verified.")
    print("\n============================================================")
    print("   Demo Completed Successfully")
    print("============================================================")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)
    run_demo()
