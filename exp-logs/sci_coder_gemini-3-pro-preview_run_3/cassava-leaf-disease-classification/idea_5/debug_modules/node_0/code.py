import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import CFG
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import CassavaClassifier
from library.engine import train_model
from library.inference import generate_submission


def run_demo():
    print("=== Starting Cassava Disease Classification Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Override CFG for a fast demonstration run
    CFG.debug = True  # Use subset of data (100 train, 50 val)
    CFG.epochs = 1  # Run only 1 epoch
    CFG.img_size = 224  # Use smaller image size for speed
    CFG.train_batch_size = 8  # Small batch size
    CFG.valid_batch_size = 8
    CFG.output_dir = "./working/demo_run"  # Custom output directory

    # Create output directory
    CFG.setup()

    # Set random seeds
    seed_everything(CFG.seed)

    print(f"    Output Directory: {CFG.output_dir}")
    print(f"    Device: {CFG.device}")
    print(f"    Debug Mode: {CFG.debug}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[2] Initializing DataLoaders...")

    # Get dataloaders
    # We disable cache loading to ensure we read from the source metadata for this demo
    train_loader, val_loader, mixup_fn = get_dataloaders(load_cached_data=False)

    # Validation: Check batch structure
    try:
        images, labels = next(iter(train_loader))
        print(f"    Train Batch - Images: {images.shape}, Labels: {labels.shape}")

        # Assertions
        assert images.shape[0] == CFG.train_batch_size
        assert images.shape[1] == 3
        assert images.shape[2] == CFG.img_size
        assert images.shape[3] == CFG.img_size

        # Check MixUp if active
        if mixup_fn is not None:
            mixed_images, mixed_labels = mixup_fn(images, labels)
            print(
                f"    MixUp Output - Images: {mixed_images.shape}, Labels: {mixed_labels.shape}"
            )
            assert mixed_labels.shape == (CFG.train_batch_size, CFG.num_classes)

    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n[3] Initializing Model...")

    # Initialize model (pretrained=False for speed/offline assurance)
    model = CassavaClassifier(
        model_name=CFG.model_name,
        pretrained=False,
        num_classes=CFG.num_classes,
        img_size=CFG.img_size,
    )
    model.to(CFG.device)

    # Validation: Check forward pass
    dummy_input = torch.randn(2, 3, CFG.img_size, CFG.img_size).to(CFG.device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"    Forward Pass Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, CFG.num_classes)

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch)...")

    # Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr)

    # Run training
    best_acc = train_model(
        model, train_loader, val_loader, optimizer, scheduler, CFG.device, mixup_fn
    )

    print(f"    Training completed. Best Validation Accuracy: {best_acc:.4f}")

    # Validation: Check if model weights are saved
    saved_weights = os.path.join(CFG.output_dir, "best_model.pth")
    if os.path.exists(saved_weights):
        print(f"    Model weights saved successfully at {saved_weights}")
    else:
        raise AssertionError(f"Model weights not found at {saved_weights}")

    # ---------------------------------------------------------
    # 5. Inference & Submission
    # ---------------------------------------------------------
    print("\n[5] Generating Submission...")

    # Generate submission using the saved model
    generate_submission(load_cached_data=False)

    # Validation: Check submission file
    submission_file = "./submission/submission.csv"
    if os.path.exists(submission_file):
        df_sub = pd.read_csv(submission_file)
        print(f"    Submission file created at {submission_file}")
        print(f"    Submission Shape: {df_sub.shape}")
        print(f"    Columns: {list(df_sub.columns)}")

        assert "image_id" in df_sub.columns
        assert "label" in df_sub.columns
        assert len(df_sub) > 0
    else:
        raise AssertionError("Submission file was not created!")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
