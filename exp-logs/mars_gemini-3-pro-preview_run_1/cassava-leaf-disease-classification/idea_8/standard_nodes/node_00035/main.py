import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
import importlib

# --- Module Reloading (Cite debug_lesson_3) ---
# Force reload of library modules to pick up Config changes in persistent environment
if "library.config" in sys.modules:
    import library.config

    importlib.reload(library.config)

    # Reload modules that import Config to update their local references
    modules_to_reload = [
        "library.utils",
        "library.augmentations",
        "library.dataset",
        "library.loss",
        "library.modeling",
        "library.engine",
        "library.inference",
    ]

    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger, save_checkpoint
from library.dataset import get_loaders, prepare_folds
from library.modeling import CassavaClassifier, get_optimizer_params
from library.engine import train_one_epoch, validate, run_swa
from library.inference import run_inference


def run():
    # --- 1. Setup & Config Overrides ---
    seed_everything(Config.seed)

    # Override Config for Fast Baseline (Target: < 2 hours)
    # We maintain Config.n_folds = 5 to ensure consistent data splitting,
    # but we will only iterate and train on Fold 0.

    # Reduced Epoch Schedule:
    # Phase 1: 2 Epochs (1 Warmup + 1 Base)
    # Phase 2: 1 Epoch (High Res Adaptation)
    # Phase 3: 2 Epochs (SWA Stabilization)
    # Total: 5 Epochs per model * 2 Models = 10 Epochs total.
    # Estimated time on A100: ~60 minutes.

    Config.p1_epochs = 2
    Config.p1_warmup_epochs = 1
    Config.p2_epochs = 1
    Config.swa_epochs = 2

    # Ensure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)

    # Setup Logger
    logger = get_logger(os.path.join(Config.output_dir, "run.log"))
    logger.info("Starting Fast Baseline Orchestration...")
    logger.info(
        f"Configuration: {Config.p1_epochs} Base Epochs, {Config.p2_epochs} FT Epochs, {Config.swa_epochs} SWA Epochs"
    )

    # Prepare Folds (ensure cached)
    # This generates the 5-fold split and saves to parquet
    prepare_folds(load_cached_data=False)

    device = Config.device

    # --- 2. Training Loop ---
    # We only train Fold 0 to save time
    folds_to_train = [0]

    for model_name in Config.model_names:
        for fold in folds_to_train:
            logger.info(
                f"\n{'='*20}\nTraining Model: {model_name} | Fold: {fold}\n{'='*20}"
            )

            # ------------------------------------------------------------------
            # Phase 1: Base Training (384x384)
            # ------------------------------------------------------------------
            logger.info(
                f"--- Phase 1: Base Training ({Config.p1_image_size}x{Config.p1_image_size}) ---"
            )

            train_loader, val_loader = get_loaders(
                fold, image_size=Config.p1_image_size, batch_size=Config.p1_batch_size
            )

            model = CassavaClassifier(model_name, pretrained=True)
            model.to(device)

            # Optimizer with LLRD
            optimizer = torch.optim.AdamW(
                get_optimizer_params(
                    model, Config.p1_lr, Config.weight_decay, Config.llrd_decay
                )
            )

            # Scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.p1_epochs * len(train_loader), eta_min=1e-6
            )

            best_acc = 0.0

            for epoch in range(Config.p1_epochs):
                # Warmup Logic
                if epoch < Config.p1_warmup_epochs:
                    logger.info("Warmup: Freezing backbone")
                    model.backbone.requires_grad_(False)
                elif epoch == Config.p1_warmup_epochs:
                    logger.info("Warmup Complete: Unfreezing backbone")
                    model.backbone.requires_grad_(True)

                loss, acc = train_one_epoch(
                    model,
                    optimizer,
                    scheduler,
                    train_loader,
                    device,
                    epoch,
                    Config.p1_accum_steps,
                )
                val_loss, val_acc = validate(model, val_loader, device)

                logger.info(
                    f"Epoch {epoch+1}/{Config.p1_epochs} - Loss: {loss:.4f} - Val Acc: {val_acc:.4f}"
                )

                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(
                        model.state_dict(),
                        True,
                        Config.output_dir,
                        f"{model_name}_fold_{fold}_stage1.pth",
                    )

            # Load best weights from Stage 1
            best_stage1_path = os.path.join(Config.output_dir, "model_best.pth")
            if os.path.exists(best_stage1_path):
                model.load_state_dict(torch.load(best_stage1_path))

            # ------------------------------------------------------------------
            # Phase 2: High-Resolution Fine-Tuning (512x512)
            # ------------------------------------------------------------------
            logger.info(
                f"--- Phase 2: Fine-Tuning ({Config.p2_image_size}x{Config.p2_image_size}) ---"
            )

            train_loader, val_loader = get_loaders(
                fold, image_size=Config.p2_image_size, batch_size=Config.p2_batch_size
            )

            # Re-initialize Optimizer with lower LR
            optimizer = torch.optim.AdamW(
                get_optimizer_params(
                    model, Config.p2_lr, Config.weight_decay, Config.llrd_decay
                )
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.p2_epochs * len(train_loader), eta_min=1e-7
            )

            for epoch in range(Config.p2_epochs):
                loss, acc = train_one_epoch(
                    model,
                    optimizer,
                    scheduler,
                    train_loader,
                    device,
                    epoch,
                    Config.p2_accum_steps,
                )
                val_loss, val_acc = validate(model, val_loader, device)
                logger.info(
                    f"FT Epoch {epoch+1}/{Config.p2_epochs} - Val Acc: {val_acc:.4f}"
                )

                # Always save the result of FT
                save_checkpoint(
                    model.state_dict(),
                    False,
                    Config.output_dir,
                    f"{model_name}_fold_{fold}_stage2.pth",
                )

            # ------------------------------------------------------------------
            # Phase 3: Stochastic Weight Averaging (SWA)
            # ------------------------------------------------------------------
            logger.info("--- Phase 3: SWA ---")

            swa_checkpoints = []
            optimizer = torch.optim.AdamW(
                get_optimizer_params(model, Config.swa_lr_start, Config.weight_decay)
            )
            # Cyclic LR
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=len(train_loader), T_mult=1, eta_min=Config.swa_lr_end
            )

            for epoch in range(Config.swa_epochs):
                loss, acc = train_one_epoch(
                    model,
                    optimizer,
                    scheduler,
                    train_loader,
                    device,
                    epoch,
                    Config.swa_accum_steps,
                )

                # Save snapshot
                ckpt_path = os.path.join(
                    Config.output_dir, f"{model_name}_fold_{fold}_swa_{epoch}.pth"
                )
                torch.save(model.state_dict(), ckpt_path)
                swa_checkpoints.append(ckpt_path)

            # Perform SWA Averaging
            model = run_swa(model, train_loader, swa_checkpoints, device)

            # Save Final Model
            final_path = os.path.join(
                Config.output_dir, f"{model_name}_fold_{fold}.pth"
            )
            torch.save(model.state_dict(), final_path)
            logger.info(f"Saved final model to {final_path}")

            # Cleanup
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

    # --- 3. Ensemble Validation ---
    logger.info("\n=== Running Ensemble Validation (Fold 0) ===")

    # Load Validation Data (Fold 0, High Res)
    _, val_loader = get_loaders(
        fold=0, image_size=Config.p2_image_size, batch_size=Config.p1_batch_size
    )

    # Collect Targets
    targets = []
    with torch.no_grad():
        for _, labels in val_loader:
            targets.append(labels.numpy())
    targets = np.concatenate(targets)

    ensemble_probs = []

    for model_name in Config.model_names:
        ckpt_path = os.path.join(Config.output_dir, f"{model_name}_fold_0.pth")
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue

        logger.info(f"Evaluating {model_name}...")
        model = CassavaClassifier(model_name, pretrained=False)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                out = model(images)
                probs.append(torch.softmax(out, dim=1).cpu().numpy())

        ensemble_probs.append(np.concatenate(probs))
        del model
        torch.cuda.empty_cache()

    if not ensemble_probs:
        logger.error("No models available for validation.")
        return

    # Average Probabilities
    avg_probs = np.mean(ensemble_probs, axis=0)
    preds = np.argmax(avg_probs, axis=1)

    # Compute Metric
    final_acc = (preds == targets).mean()
    print(f"Final Validation Metric: {final_acc}")

    # --- 4. Failure Analysis ---
    logger.info("\n=== Failure Analysis ===")
    try:
        # Load Fold Metadata
        df_folds = pd.read_parquet(os.path.join(Config.output_dir, "folds.parquet"))
        df_val = df_folds[df_folds["fold"] == 0].reset_index(drop=True)

        if len(df_val) == len(preds):
            df_val["pred"] = preds
            df_val["error"] = (df_val["label"] != df_val["pred"]).astype(int)

            # Compute File Size
            file_sizes = []
            for _, row in df_val.iterrows():
                try:
                    p = os.path.join(Config.input_root, row["file_path"])
                    file_sizes.append(os.path.getsize(p))
                except:
                    file_sizes.append(0)
            df_val["file_size"] = file_sizes

            # Correlation
            corr = df_val[["error", "file_size", "label"]].corr()["error"]
            print("Error Correlation Matrix:")
            print(corr)
        else:
            logger.warning("Validation size mismatch. Skipping detailed analysis.")
    except Exception as e:
        logger.error(f"Failure analysis failed: {e}")

    # --- 5. Submission ---
    threshold = 0.9076101468624833
    if final_acc > threshold:
        logger.info(f"Metric {final_acc} > {threshold}. Generating Submission...")
        # run_inference automatically handles the ensemble of available checkpoints
        run_inference()
    else:
        logger.info(f"Metric {final_acc} <= {threshold}. Skipping Submission.")


if __name__ == "__main__":
    run()
