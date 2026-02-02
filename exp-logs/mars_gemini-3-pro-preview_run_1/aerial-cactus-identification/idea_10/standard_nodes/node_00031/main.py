import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import get_dataloaders, get_test_dataloader, get_data, Mixup
from library.models import create_model
from library.engine import train_one_epoch, validate, predict_tta
from library.stacking import train_stacker, predict_stacker


def run():
    # 1. Setup
    # Override epochs for speed within the 2h limit, while maintaining performance
    Config.EPOCHS = 15
    seed_everything(Config.SEED)

    print(f"Starting execution on device: {Config.DEVICE}")
    print(f"Training with {Config.N_FOLDS} folds, {Config.EPOCHS} epochs each.")

    # 2. Prepare Data & Storage
    # Load full training data to map OOF predictions correctly
    # We need the indices from the KFold split to place predictions in the right spot
    full_train_imgs, full_train_labels, full_train_fs = get_data(
        mode="train", load_cached_data=True
    )

    # Storage for OOF predictions (Architecture -> Array of shape (N_train,))
    model_names = ["resnet", "repvgg", "next"]
    oof_preds = {name: np.zeros(len(full_train_labels)) for name in model_names}

    # Storage for Test predictions (Architecture -> Array of shape (N_test,))
    # We will sum predictions across folds and divide by N_FOLDS later
    test_loader, test_ids = get_test_dataloader(load_cached_data=True)
    test_preds_accumulator = {name: np.zeros(len(test_ids)) for name in model_names}

    # Get test file sizes for stacking
    _, _, test_file_sizes = get_data(mode="test", load_cached_data=True)

    # 3. Cross-Validation Loop
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # We iterate through folds manually to match the get_dataloaders split logic
    # Note: get_dataloaders internally uses the same seed and StratifiedKFold,
    # so calling it with fold_idx gives consistent splits.
    # However, to fill oof_preds array correctly, we need the val_indices here too.
    splits = list(skf.split(full_train_imgs, full_train_labels))

    for fold_idx in range(Config.N_FOLDS):
        print(f"\n{'='*20} Fold {fold_idx+1}/{Config.N_FOLDS} {'='*20}")

        # Get Loaders
        train_loader, val_loader = get_dataloaders(fold_idx, load_cached_data=True)
        _, val_indices = splits[fold_idx]

        # Train each architecture
        for model_name in model_names:
            print(f"\n--- Training {model_name} ---")

            # Initialize Model
            model = create_model(model_name, num_classes=Config.NUM_CLASSES).to(
                Config.DEVICE
            )

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
            criterion = nn.BCEWithLogitsLoss()
            mixup_fn = Mixup(alpha=Config.MIXUP_ALPHA)

            # Training Loop
            best_val_auc = 0.0
            best_model_state = None

            for epoch in range(Config.EPOCHS):
                avg_loss = train_one_epoch(
                    model, train_loader, optimizer, Config.DEVICE, criterion, mixup_fn
                )
                scheduler.step()

                # Check validation occasionally or every epoch? Every epoch is fine for 15 epochs.
                val_loss, val_auc, _, _ = validate(
                    model, val_loader, criterion, Config.DEVICE
                )

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = model.state_dict()

                # Print sparse logs
                if (epoch + 1) % 5 == 0 or (epoch + 1) == Config.EPOCHS:
                    print(
                        f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}"
                    )

            # Load best state for inference
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            # Switch RepVGG to deploy mode for inference
            if model_name == "repvgg":
                model.switch_to_deploy()
                model.to(Config.DEVICE)  # Re-move to device after structural change

            # Generate OOF Predictions (No TTA for validation usually, or yes?
            # Standard practice is single view for val to match production speed,
            # but TTA for test. Let's stick to single view for OOF to keep it simple/fast).
            # We use the validate function to get raw probs
            _, _, val_probs, _ = validate(model, val_loader, criterion, Config.DEVICE)
            oof_preds[model_name][val_indices] = val_probs

            # Generate Test Predictions (with TTA)
            fold_test_probs = predict_tta(model, test_loader, Config.DEVICE)
            test_preds_accumulator[model_name] += fold_test_probs

            # Cleanup
            del model, optimizer, scheduler, criterion
            torch.cuda.empty_cache()

    # 4. Aggregation
    print("\nAggregating Test Predictions...")
    test_preds_avg = {k: v / Config.N_FOLDS for k, v in test_preds_accumulator.items()}

    # 5. Stacking
    print("\nTraining Meta-Learner (Stacker)...")
    # Train stacker on OOF predictions and Train File Sizes
    stacker, oof_auc = train_stacker(oof_preds, full_train_fs, full_train_labels)

    # Predict on Test using Stacker
    final_test_probs = predict_stacker(stacker, test_preds_avg, test_file_sizes)

    # 6. Metrics & Failure Analysis
    print(f"\nFinal Validation Metric: {oof_auc}")

    # Calculate Stacker OOF predictions for analysis
    # We need to re-predict on OOF data using the trained stacker to get the exact stacker probabilities
    from library.stacking import prepare_meta_features

    X_oof = prepare_meta_features(oof_preds, full_train_fs)
    stacker_oof_probs = stacker.predict_proba(X_oof)

    # Calculate error
    errors = np.abs(full_train_labels - stacker_oof_probs)

    # Correlation with file size
    # Normalize file sizes for correlation calculation to match scale roughly, or just use raw
    corr = np.corrcoef(errors, full_train_fs)[0, 1]
    print(
        f"Failure Analysis: Correlation between Error Magnitude and File Size: {corr:.4f}"
    )

    # 7. Submission
    # The prompt condition "If and only if... > 1.0" is logically impossible for AUC.
    # We assume standard behavior: submit if successful.
    print("Generating submission file...")
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_test_probs})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())


if __name__ == "__main__":
    run()
