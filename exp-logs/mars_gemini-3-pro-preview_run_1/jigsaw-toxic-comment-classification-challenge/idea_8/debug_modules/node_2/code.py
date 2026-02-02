import os
import torch
import pandas as pd
import numpy as np
import warnings
import gc
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import (
    get_tokenizer,
    get_data,
    prepare_loaders,
    prepare_mlm_loader,
    prepare_test_loader,
)
from library.model import CustomDeberta
from library.awp import AWP
from library.engine import train_mlm, train_fn, valid_fn, inference_fn

# Suppress warnings
warnings.filterwarnings("ignore")


def run_demo():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("[Demo] Setting up configuration...")

    # Override Config for fast demonstration
    Config.debug = True
    Config.debug_sample_size = 64  # Small sample for speed
    Config.epochs = 1
    Config.n_folds = 1  # Run only one fold
    Config.train_batch_size = 4
    Config.batch_size = 4
    Config.mlm_batch_size = 4
    Config.mlm_epochs = 1
    Config.awp_start_epoch = 0  # Enable AWP immediately for demo
    Config.num_workers = 0  # Avoid multiprocessing overhead in demo
    Config.working_dir = "./working/demo_run"
    Config.mlm_model_dir = os.path.join(Config.working_dir, "mlm_backbone")

    # Create working directories
    os.makedirs(Config.working_dir, exist_ok=True)
    os.makedirs(Config.mlm_model_dir, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.seed)

    device = Config.device
    print(f"[Demo] Device: {device}")

    # =========================================================================
    # 2. Data Preparation
    # =========================================================================
    print("[Demo] Loading Tokenizer and Data...")
    tokenizer = get_tokenizer()

    # Load data (Debug mode will sample 64 rows)
    train_df, test_df = get_data(load_cached_data=False)

    # Validations
    assert (
        len(train_df) == Config.debug_sample_size
    ), f"Train size mismatch: {len(train_df)}"
    assert (
        len(test_df) == Config.debug_sample_size
    ), f"Test size mismatch: {len(test_df)}"
    print(
        f"[Demo] Data loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}"
    )

    # =========================================================================
    # 3. Domain-Adaptive Pre-training (MLM)
    # =========================================================================
    print("[Demo] Running MLM Pre-training...")

    # Prepare MLM Loader
    mlm_loader = prepare_mlm_loader(tokenizer, load_cached_data=False)

    # Run MLM Training (Saves model to Config.mlm_model_dir)
    train_mlm(mlm_loader, device)

    # Verify MLM model was saved
    assert os.path.exists(Config.mlm_model_dir), "MLM model directory not created."
    # Check for config.json or pytorch_model.bin/model.safetensors
    has_model_file = any(
        f.endswith(".bin") or f.endswith(".safetensors") or f == "config.json"
        for f in os.listdir(Config.mlm_model_dir)
    )
    assert has_model_file, "MLM model files not found."
    print("[Demo] MLM Pre-training complete.")

    # =========================================================================
    # 4. Supervised Fine-Tuning (SFT)
    # =========================================================================
    print("[Demo] Starting Supervised Fine-Tuning...")

    # We only run Fold 0 for the demo
    fold = 0

    # Prepare DataLoaders
    train_loader, val_loader = prepare_loaders(fold, train_df, tokenizer, debug=True)

    # Initialize Model
    # We load the backbone from the MLM step we just finished
    model = CustomDeberta(pretrained_path=Config.mlm_model_dir)
    model.to(device)

    # Optimization Setup
    optimizer = AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Scheduler
    num_train_steps = len(train_loader) * Config.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        total_steps=num_train_steps,
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # Adversarial Weight Perturbation
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
    )

    # Training Loop
    criterion = None  # Loss is calculated inside the model forward pass

    for epoch in range(Config.epochs):
        print(f"[Demo] Epoch {epoch + 1}/{Config.epochs}")

        # Train
        avg_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, device, epoch, awp=awp
        )
        print(f"[Demo] Train Loss: {avg_loss:.4f}")

        # Validation
        val_loss, val_score = valid_fn(val_loader, model, device)
        print(f"[Demo] Val Loss: {val_loss:.4f}, Val AUC: {val_score:.4f}")

        # Assertions
        assert not np.isnan(avg_loss), "Training loss is NaN"
        assert 0 <= val_score <= 1, f"Validation AUC out of range: {val_score}"

    # =========================================================================
    # 5. Inference
    # =========================================================================
    print("[Demo] Running Inference on Test Set...")

    test_loader = prepare_test_loader(test_df, tokenizer)
    preds = inference_fn(test_loader, model, device)

    # Verify predictions
    assert preds.shape == (
        len(test_df),
        Config.num_classes,
    ), f"Prediction shape mismatch. Expected {(len(test_df), Config.num_classes)}, got {preds.shape}"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions out of probability range [0, 1]"

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    print("[Demo] Generating Submission File...")

    submission_df = pd.DataFrame(preds, columns=Config.target_cols)
    submission_df.insert(0, "id", test_df["id"])

    sub_path = os.path.join(Config.working_dir, "submission.csv")
    submission_df.to_csv(sub_path, index=False)

    assert os.path.exists(sub_path), "Submission file not created."
    print(f"[Demo] Submission saved to {sub_path}")
    print("[Demo] Head of submission:")
    print(submission_df.head())

    # Cleanup
    del model, optimizer, scheduler, awp, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    print("\n[Demo] Completed successfully.")


if __name__ == "__main__":
    run_demo()
