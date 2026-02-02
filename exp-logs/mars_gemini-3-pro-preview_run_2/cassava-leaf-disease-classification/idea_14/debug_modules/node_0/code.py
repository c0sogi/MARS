import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import library components
from library.config import CFG
from library.utils import seed_everything, get_logger
from library.data import load_metadata, create_loaders, Mixup
from library.network import get_model, ModelEMA
from library.trainer import train_one_epoch, validate
from library.inference import inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print(">>> Starting Cassava Leaf Disease Classification Demo")

    # ==========================================
    # 1. Configuration Overrides for Demo Speed
    # ==========================================
    print("\n[1] Configuring environment...")

    # Set paths to a demo directory to avoid interfering with real experiments
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override CFG settings
    CFG.working_dir = DEMO_DIR
    CFG.output_dir = DEMO_DIR
    CFG.submission_csv = os.path.join(DEMO_DIR, "submission.csv")

    # Enable debug mode to load a tiny subset of data
    CFG.debug = True
    CFG.debug_sample_size = 100  # Small sample for rapid execution

    # Reduce training complexity
    CFG.model_name = "resnet18"  # Use a lighter model for the demo
    CFG.n_folds = 2  # Simulate a 2-fold setup
    CFG.p1_epochs = 1  # Only 1 epoch
    CFG.p2_epochs = 0  # Skip phase 2
    CFG.p1_batch_size = 16
    CFG.target_batch_size = 16  # No gradient accumulation needed for demo
    CFG.num_workers = 2  # Reduce overhead

    # Align Phase 2 size with Phase 1 for inference consistency in this quick test
    # (Since we skip Phase 2 training, we infer on Phase 1 size)
    CFG.p2_img_size = CFG.p1_img_size

    # Seed everything
    seed_everything(CFG.seed)

    # Setup Logger
    logger = get_logger(os.path.join(CFG.working_dir, "demo.log"))
    logger.info("Configuration configured for fast demonstration.")

    # ==========================================
    # 2. Data Loading & Verification
    # ==========================================
    logger.info("\n[2] Loading Data...")
    train_df, val_df, test_df = load_metadata()

    # Verify we got the debug size (or less if original data is smaller)
    logger.info(
        f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}"
    )
    assert len(train_df) <= CFG.debug_sample_size
    assert len(val_df) <= CFG.debug_sample_size

    # Create Loaders
    train_loader, val_loader = create_loaders(
        train_df, val_df, CFG.p1_img_size, CFG.p1_batch_size
    )

    # Verify Loader Output
    images, labels = next(iter(train_loader))
    logger.info(
        f"Batch loaded. Image shape: {images.shape}, Label shape: {labels.shape}"
    )

    assert images.shape == (CFG.p1_batch_size, 3, CFG.p1_img_size, CFG.p1_img_size)
    assert labels.shape == (CFG.p1_batch_size,)
    assert labels.dtype == torch.long

    # Verify Mixup
    logger.info("Verifying Mixup augmentation...")
    mixup_fn = Mixup(num_classes=CFG.num_classes)
    mixed_x, mixed_y = mixup_fn(images, labels)

    assert mixed_x.shape == images.shape
    assert mixed_y.shape == (CFG.p1_batch_size, CFG.num_classes)
    logger.info("Mixup shapes correct.")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    logger.info("\n[3] Initializing Model...")
    # Use pretrained=False to ensure it runs without downloading large weights if internet is restricted
    model = get_model(pretrained=False)
    model_ema = ModelEMA(model)

    # Verify Model Output
    with torch.no_grad():
        output = model(images.to(CFG.device))
    assert output.shape == (CFG.p1_batch_size, CFG.num_classes)
    logger.info("Model forward pass successful.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    logger.info("\n[4] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    loss, acc = train_one_epoch(
        epoch=1,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=CFG.device,
        model_ema=model_ema,
        logger=logger,
    )

    logger.info(f"Epoch 1 Completed. Loss: {loss:.4f}, Acc: {acc:.4f}")
    assert not np.isnan(loss), "Training loss is NaN"

    # ==========================================
    # 5. Validation Demonstration
    # ==========================================
    logger.info("\n[5] Running Validation...")
    val_loss, val_acc = validate(model_ema.module, val_loader, CFG.device, logger)
    logger.info(f"Validation Completed. Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # ==========================================
    # 6. Inference Demonstration
    # ==========================================
    logger.info("\n[6] Preparing for Inference...")

    # Save the current model as "best_model_fold_0.pth" and "best_model_fold_1.pth"
    # to simulate a completed cross-validation training
    fold_0_path = os.path.join(CFG.output_dir, "best_model_fold_0.pth")
    fold_1_path = os.path.join(CFG.output_dir, "best_model_fold_1.pth")

    torch.save(model_ema.module.state_dict(), fold_0_path)
    shutil.copy(fold_0_path, fold_1_path)

    assert os.path.exists(fold_0_path)
    assert os.path.exists(fold_1_path)

    logger.info("Running Inference Module...")

    # Run the full inference routine provided in the library
    # This will load test metadata, load the models we just saved, predict, and save submission.csv
    inference()

    # Verify Submission
    submission_path = CFG.submission_csv
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    logger.info(f"Submission generated with {len(sub_df)} rows.")

    # Check against test metadata length
    assert len(sub_df) == len(test_df)
    assert "image_id" in sub_df.columns
    assert "label" in sub_df.columns

    # Also check the specific submission directory required by the task
    final_sub_path = "./submission/submission.csv"
    assert os.path.exists(
        final_sub_path
    ), "Final submission file not found in ./submission"

    logger.info(">>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
