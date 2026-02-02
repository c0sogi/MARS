import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import soundfile as sf
from sklearn.metrics import roc_auc_score

# Import library modules
from library.config import Config
from library.utils import seed_everything, setup_logger, calculate_roc_auc
from library.data import get_dataloaders
from library.models import WhaleClassifier
from library.engine import Trainer, predict
from library.pseudo_labeling import infer_on_test, generate_pseudo_labels
from library.ensemble import (
    load_oof_features,
    load_test_features,
    train_meta_learner,
    generate_submission,
)


def analyze_failures(oof_df, config):
    """
    Performs failure analysis on the OOF predictions by correlating errors with audio metadata.
    """
    print("\n--- Failure Analysis ---")

    # Calculate Error Magnitude
    # If meta_prob exists, use it; otherwise average base models (fallback)
    if "meta_prob" in oof_df.columns:
        preds = oof_df["meta_prob"]
    else:
        base_cols = [c for c in oof_df.columns if c not in ["label", "meta_prob"]]
        preds = oof_df[base_cols].mean(axis=1)

    targets = oof_df["label"]
    errors = np.abs(targets - preds)

    # Reconstruct metadata to map OOF indices to file paths
    train_meta = pd.read_csv(config.TRAIN_CSV)
    val_meta = pd.read_csv(config.VAL_CSV)
    full_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    print("Extracting audio features for correlation analysis (Sampled)...")
    durations = []
    rms_values = []

    # Sample 2000 files for quick analysis to respect time limits
    sample_size = min(2000, len(full_meta))
    sample_indices = np.random.choice(len(full_meta), size=sample_size, replace=False)

    for idx in sample_indices:
        row = full_meta.iloc[idx]
        path = os.path.join(config.INPUT_ROOT, row["file_path"])
        try:
            # Quick read of audio stats
            data, sr = sf.read(path)
            durations.append(len(data) / sr)
            rms_values.append(np.sqrt(np.mean(data**2)))
        except Exception:
            durations.append(0)
            rms_values.append(0)

    # Calculate correlation on the subset
    subset_errors = errors.iloc[sample_indices].values

    corr_dur = np.corrcoef(subset_errors, durations)[0, 1]
    corr_rms = np.corrcoef(subset_errors, rms_values)[0, 1]

    print(f"Correlation between Error and Duration: {corr_dur:.4f}")
    print(f"Correlation between Error and RMS Energy: {corr_rms:.4f}")


def run():
    # 1. Configuration
    # Override num_epochs to 3 for a fast baseline execution as requested
    config = Config(num_epochs=3, batch_size=128)
    config.create_directories()

    # Logger
    logger = setup_logger(os.path.join(config.WORKING_DIR, "train.log"))
    seed_everything(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # ROUND 1: Supervised Training
    # ==========================================
    print("\n=== Round 1: Supervised Training ===")

    round1_checkpoints = []  # List of (model_name, path)

    for model_name in config.MODELS:
        for fold in range(config.NUM_FOLDS):
            print(f"\nTraining {model_name} - Fold {fold} (Round 1)")

            # Check if checkpoint exists to skip (resumption capability)
            ckpt_path_auc = os.path.join(
                config.CHECKPOINT_DIR, f"{model_name}_fold_{fold}_best_auc.pth"
            )

            # DataLoaders
            train_loader, val_loader = get_dataloaders(
                config, fold=fold, mode="train", load_cached_data=True
            )

            # Model
            model = WhaleClassifier(
                model_name, pretrained=True, in_channels=config.IN_CHANNELS
            )
            model.to(device)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.NUM_EPOCHS
            )

            # Trainer
            trainer = Trainer(
                config,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                device,
                fold,
                model_name,
                logger,
            )
            trainer.fit()

            # Store checkpoint path for pseudo-labeling
            round1_checkpoints.append((model_name, ckpt_path_auc))

            # Free memory
            del model, optimizer, scheduler, trainer, train_loader, val_loader
            torch.cuda.empty_cache()

    # ==========================================
    # Pseudo-Labeling
    # ==========================================
    print("\n=== Generating Pseudo-Labels ===")

    # Infer on Test Set using Round 1 models
    test_probs = infer_on_test(
        config, round1_checkpoints, device, load_cached_preds=False
    )

    # Generate Pseudo Labels
    pseudo_df = generate_pseudo_labels(config, test_probs)

    # ==========================================
    # ROUND 2: Semi-Supervised Training
    # ==========================================
    print("\n=== Round 2: Semi-Supervised Training ===")

    for model_name in config.MODELS:
        for fold in range(config.NUM_FOLDS):
            print(f"\nTraining {model_name} - Fold {fold} (Round 2)")

            # DataLoaders with Pseudo Data
            # Note: val_loader remains the original pure validation set
            train_loader, val_loader = get_dataloaders(
                config,
                fold=fold,
                mode="train",
                pseudo_df=pseudo_df,
                load_cached_data=True,
            )

            # Model - Retrain from scratch (pretrained=True loads ImageNet weights)
            model = WhaleClassifier(
                model_name, pretrained=True, in_channels=config.IN_CHANNELS
            )
            model.to(device)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.NUM_EPOCHS
            )

            trainer = Trainer(
                config,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                device,
                fold,
                model_name,
                logger,
            )
            trainer.fit()

            # --- Generate & Save OOF and Test Predictions (Round 2) ---
            # We need predictions for both Best AUC and Best Loss checkpoints for the ensemble

            # Pre-load Test Loader once to save overhead? No, keep it simple inside loop to manage memory.
            test_loader = get_dataloaders(config, mode="test", load_cached_data=True)

            for metric in ["best_auc", "best_loss"]:
                ckpt_path = os.path.join(
                    config.CHECKPOINT_DIR, f"{model_name}_fold_{fold}_{metric}.pth"
                )

                if os.path.exists(ckpt_path):
                    # Load Checkpoint
                    checkpoint = torch.load(ckpt_path, map_location=device)
                    state_dict = (
                        checkpoint["state_dict"]
                        if "state_dict" in checkpoint
                        else checkpoint
                    )
                    # Remove module. prefix if present
                    state_dict = {
                        k.replace("module.", ""): v for k, v in state_dict.items()
                    }
                    model.load_state_dict(state_dict)
                    model.eval()

                    # 1. OOF Predictions (on Validation Set)
                    oof_preds = predict(model, val_loader, device)
                    oof_save_path = os.path.join(
                        config.OOF_DIR, f"{model_name}_{metric}_fold_{fold}.npy"
                    )
                    np.save(oof_save_path, oof_preds)

                    # 2. Test Predictions (for Stacking)
                    test_preds = predict(model, test_loader, device)
                    test_save_path = os.path.join(
                        config.PREDS_DIR, f"{model_name}_{metric}_fold_{fold}.npy"
                    )
                    np.save(test_save_path, test_preds)

            # Free memory
            del (
                model,
                optimizer,
                scheduler,
                trainer,
                train_loader,
                val_loader,
                test_loader,
            )
            torch.cuda.empty_cache()

    # ==========================================
    # Ensembling & Submission
    # ==========================================
    print("\n=== Ensembling ===")

    # Load Features (Force reload to get Round 2 data)
    X_oof = load_oof_features(config, load_cached_data=False)
    X_test = load_test_features(config, load_cached_data=False)

    # Train Meta-Learner
    meta_learner = train_meta_learner(X_oof)

    # Calculate Final Validation Metric (OOF AUC)
    feature_cols = [c for c in X_oof.columns if c != "label"]
    oof_probs = meta_learner.predict_proba(X_oof[feature_cols].values)[:, 1]
    final_auc = calculate_roc_auc(X_oof["label"].values, oof_probs)

    print(f"Final Validation Metric: {final_auc}")

    # Add meta_prob to X_oof for failure analysis
    X_oof["meta_prob"] = oof_probs

    # Failure Analysis
    analyze_failures(X_oof, config)

    # Submission
    threshold = 0.9998881660199745
    if final_auc > threshold:
        generate_submission(config, meta_learner, X_test)
    else:
        print(
            f"Validation metric {final_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
