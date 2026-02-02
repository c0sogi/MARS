import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders, get_dataframes
from library.model import AppleDiseaseModel
from library.engine import train_one_epoch, validate, calculate_pos_weights


def run_training():
    """
    Executes the training pipeline for the Apple Disease Detection task.
    Iterates over defined models and folds, implements SWA, and saves the best checkpoints.
    """
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # Load full dataframe to assist with weight calculation
    # get_dataframes handles the caching and splitting logic
    full_train_df, _ = get_dataframes(load_cached_data=True)

    device = torch.device(Config.DEVICE)

    # Iterate over model configurations
    for model_config in Config.MODELS:
        model_name = model_config["model_name"]
        fold_indices = model_config["fold_indices"]
        img_size = model_config["img_size"]
        accum_steps = model_config.get("accum_steps", 1)

        print(f"\n{'='*40}")
        print(f"Training Model: {model_name}")
        print(f"Image Size: {img_size}")
        print(f"Batch Size: {model_config['batch_size']}")
        print(f"Accumulation Steps: {accum_steps}")
        print(f"{'='*40}")

        for fold in fold_indices:
            print(f"\n--- Fold {fold} ---")

            # 1. Data Loaders
            train_loader, val_loader, _ = get_loaders(
                fold=fold, model_config=model_config, load_cached_data=True
            )

            # 2. Class Weights
            # Filter the training data for this fold to calculate weights
            train_df_fold = full_train_df[full_train_df["fold"] != fold].reset_index(
                drop=True
            )

            if Config.USE_POS_WEIGHTS:
                pos_weights = calculate_pos_weights(train_df_fold, device)
                print(f"Positive Class Weights: {pos_weights.cpu().numpy()}")
            else:
                pos_weights = None

            # 3. Model Initialization
            model = AppleDiseaseModel(model_name=model_name, pretrained=True)
            model.to(device)

            # 4. Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
            )

            scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)

            # Standard Scheduler: Cosine Annealing
            # Runs until SWA starts or until end if SWA is disabled
            scheduler_epochs = (
                Config.SWA_START_EPOCH if Config.USE_SWA else Config.EPOCHS
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=scheduler_epochs, eta_min=Config.MIN_LR
            )

            # 5. SWA Initialization
            swa_model = None
            swa_scheduler = None
            if Config.USE_SWA:
                swa_model = AveragedModel(model)
                swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

            # Tracking
            best_auc = 0.0
            best_model_path = Config.get_model_path(model_name, fold)

            for epoch in range(Config.EPOCHS):
                # Determine if we are in SWA phase
                is_swa_phase = Config.USE_SWA and (epoch >= Config.SWA_START_EPOCH)

                # Train
                train_loss = train_one_epoch(
                    model,
                    optimizer,
                    train_loader,
                    device,
                    epoch,
                    pos_weights,
                    scaler=scaler,
                    accum_steps=accum_steps,
                )

                # Scheduler Step
                if is_swa_phase:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                    lr_curr = swa_scheduler.get_last_lr()[0]
                else:
                    scheduler.step()
                    lr_curr = scheduler.get_last_lr()[0]

                # Validate
                # We validate the current stochastic model to monitor progress
                val_loss, val_auc = validate(model, val_loader, device, pos_weights)

                print(
                    f"Epoch {epoch+1}/{Config.EPOCHS} | "
                    f"LR: {lr_curr:.2e} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Val AUC: {val_auc:.15f}"
                )

                # Save best standard model
                # We track this even if using SWA, in case SWA fails or for comparison
                if val_auc > best_auc:
                    best_auc = val_auc
                    # If we are NOT in SWA phase yet, this is the best model so far.
                    # If we ARE in SWA phase, the 'model' is exploring high entropy areas,
                    # so its individual performance might fluctuate, but we still save it if it peaks.
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  -> New Best Model Saved! AUC: {best_auc:.15f}")

            # 6. Finalize SWA
            if Config.USE_SWA:
                print("\nFinalizing SWA Model...")
                # Update BatchNorm statistics for the averaged model using training data
                update_bn(train_loader, swa_model, device=device)

                # Validate SWA Model
                swa_loss, swa_auc = validate(swa_model, val_loader, device, pos_weights)
                print(f"SWA Model | Val Loss: {swa_loss:.6f} | Val AUC: {swa_auc:.15f}")

                # Save SWA model
                # We overwrite the best model path with the SWA model as per strategy
                # to ensure the ensemble uses the converged, robust solution.
                print(f"Saving SWA model to {best_model_path}")
                # Save the module state dict to be compatible with AppleDiseaseModel loading
                torch.save(swa_model.module.state_dict(), best_model_path)

            # Clean up to save memory
            del model, optimizer, scheduler, train_loader, val_loader, scaler
            if swa_model:
                del swa_model, swa_scheduler
            torch.cuda.empty_cache()

    print("\nTraining Complete.")
