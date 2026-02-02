import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.modeling import get_model, verify_initialization
from library.training import train_one_epoch, validate, predict


class AppleDiseaseWorkflow:
    def __init__(self):
        self.device = Config.DEVICE
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def run_phase1(self):
        """
        Phase 1: Proxy Calibration (5-Fold CV).
        Determines the Global Optimal Epoch and TTA strategy.
        """
        print("=" * 30)
        print("PHASE 1: Proxy Calibration (5-Fold CV)")
        print("=" * 30)

        val_auc_history = np.zeros((Config.N_FOLDS, Config.EPOCHS))

        # Storage for OOF analysis to decide on TTA
        oof_preds_no_tta = []
        oof_preds_tta = []
        oof_targets = []

        for fold in range(Config.N_FOLDS):
            print(f"\n--- Fold {fold+1}/{Config.N_FOLDS} ---")
            # Ensure reproducibility per fold
            seed_everything(Config.SEED)

            # Data Loading
            train_loader, val_loader, _, class_weights = get_dataloaders(
                fold_idx=fold, phase="phase1"
            )
            class_weights = class_weights.to(self.device)

            # Model & Setup
            model = get_model(device=self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            # Scheduler synchronized with total budget
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.ETA_MIN
            )

            # Initialization Verification
            # Use unweighted loss for verification as initialization targets natural priors
            verify_initialization(
                model, train_loader, nn.CrossEntropyLoss(), self.device
            )

            # Training Loop
            best_fold_auc = -1.0
            best_model_path = os.path.join(
                self.models_dir, f"phase1_best_fold_{fold}.pth"
            )

            for epoch in range(Config.EPOCHS):
                train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, self.device
                )
                val_loss, val_auc = validate(model, val_loader, criterion, self.device)
                scheduler.step()

                val_auc_history[fold, epoch] = val_auc
                print(
                    f"Fold {fold+1} Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val AUC: {val_auc:.6f}"
                )

                # Save best model for TTA check
                if val_auc > best_fold_auc:
                    best_fold_auc = val_auc
                    torch.save(model.state_dict(), best_model_path)

            # TTA Evaluation for this fold using the best checkpoint
            print(f"Evaluating TTA for Fold {fold+1}...")
            model.load_state_dict(torch.load(best_model_path))

            # Get targets from val_loader
            fold_targets = []
            for _, labels in val_loader:
                fold_targets.append(labels.numpy())
            fold_targets = np.concatenate(fold_targets, axis=0)
            oof_targets.append(fold_targets)

            # Predict No TTA
            _, preds_no = predict(model, val_loader, self.device, use_tta=False)
            oof_preds_no_tta.append(preds_no)

            # Predict TTA
            _, preds_yes = predict(model, val_loader, self.device, use_tta=True)
            oof_preds_tta.append(preds_yes)

        # 1. Determine Global Optimal Epoch
        mean_auc_per_epoch = np.mean(val_auc_history, axis=0)
        optimal_epoch_idx = np.argmax(mean_auc_per_epoch)
        optimal_epoch = int(optimal_epoch_idx + 1)

        print("\nPhase 1 Analysis:")
        print(f"Mean AUC per epoch: {mean_auc_per_epoch}")
        print(
            f"Global Optimal Epoch (E_opt): {optimal_epoch} (Peak AUC: {mean_auc_per_epoch[optimal_epoch_idx]:.6f})"
        )

        # 2. Determine TTA Strategy
        all_targets = np.concatenate(oof_targets, axis=0)
        all_preds_no = np.concatenate(oof_preds_no_tta, axis=0)
        all_preds_yes = np.concatenate(oof_preds_tta, axis=0)

        auc_no_tta = calculate_metric(all_targets, all_preds_no)
        auc_yes_tta = calculate_metric(all_targets, all_preds_yes)

        print(f"OOF AUC (No TTA): {auc_no_tta:.6f}")
        print(f"OOF AUC (With TTA): {auc_yes_tta:.6f}")

        use_tta = False
        if auc_yes_tta > auc_no_tta:
            print("Decision: ENABLE Test-Time Augmentation (TTA).")
            use_tta = True
        else:
            print("Decision: DISABLE Test-Time Augmentation (TTA).")
            use_tta = False

        return optimal_epoch, use_tta

    def run_phase2(self, optimal_epoch):
        """
        Phase 2: Production Training (Seed Ensemble).
        Trains 5 models on 100% data for exactly optimal_epoch.
        """
        print("\n" + "=" * 30)
        print("PHASE 2: Production Training (Seed Ensemble)")
        print("=" * 30)
        print(
            f"Training {len(Config.PHASE2_SEEDS)} models on 100% data for {optimal_epoch} epochs."
        )

        for i, seed in enumerate(Config.PHASE2_SEEDS):
            print(
                f"\n--- Training Seed Model {i+1}/{len(Config.PHASE2_SEEDS)} (Seed: {seed}) ---"
            )
            seed_everything(seed)

            # Load full dataset
            train_loader, _, _, class_weights = get_dataloaders(phase="phase2")
            class_weights = class_weights.to(self.device)

            model = get_model(device=self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            # Use original T_0 to match Phase 1 dynamics, but stop early
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.ETA_MIN
            )

            # Verify initialization before burning compute
            # Use unweighted loss for verification as initialization targets natural priors
            verify_initialization(
                model, train_loader, nn.CrossEntropyLoss(), self.device
            )

            for epoch in range(optimal_epoch):
                loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, self.device
                )
                scheduler.step()
                print(
                    f"Seed {seed} Epoch {epoch+1}/{optimal_epoch} - Train Loss: {loss:.6f}"
                )

            # Save Final Model
            save_path = os.path.join(self.models_dir, f"phase2_model_seed_{seed}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    def generate_submission(self, use_tta):
        """
        Generates the final submission by ensembling Phase 2 models.
        """
        print("\n" + "=" * 30)
        print("GENERATING SUBMISSION")
        print("=" * 30)

        # Get test loader
        _, _, test_loader, _ = get_dataloaders(phase="phase2")

        ensemble_preds = []
        ids = []

        for seed in Config.PHASE2_SEEDS:
            model_path = os.path.join(self.models_dir, f"phase2_model_seed_{seed}.pth")
            print(f"Loading model from {model_path}...")

            model = get_model(device=self.device)
            model.load_state_dict(torch.load(model_path))

            # Predict
            current_ids, preds = predict(
                model, test_loader, self.device, use_tta=use_tta
            )
            ensemble_preds.append(preds)

            if not ids:
                ids = current_ids

        # Average predictions
        avg_preds = np.mean(ensemble_preds, axis=0)

        # Create DataFrame
        df_sub = pd.DataFrame(
            {
                "image_id": ids,
                "healthy": avg_preds[:, 0],
                "multiple_diseases": avg_preds[:, 1],
                "rust": avg_preds[:, 2],
                "scab": avg_preds[:, 3],
            }
        )

        # Ensure column order matches sample submission
        cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
        df_sub = df_sub[cols]

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(df_sub.head())


def run_workflow():
    """
    Main entry point to execute the full workflow.
    """
    workflow = AppleDiseaseWorkflow()

    # Step 1: Phase 1 (Calibration)
    optimal_epoch, use_tta = workflow.run_phase1()

    # Step 2: Phase 2 (Production Training)
    workflow.run_phase2(optimal_epoch)

    # Step 3: Submission
    workflow.generate_submission(use_tta)
