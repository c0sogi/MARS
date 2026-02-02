import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, calculate_metric, rank_normalize
from library.dataset import load_data, get_loaders, get_test_loader
from library.model import DiseaseClassifier


class SWATrainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        seed_everything(Config.SEED)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def get_class_weights(self, df):
        """
        Calculates positive class weights for BCEWithLogitsLoss based on imbalance.
        """
        if "target_rust" not in df.columns:
            return None

        pos_rust = df["target_rust"].sum()
        pos_scab = df["target_scab"].sum()
        total = len(df)

        # Inverse frequency weights
        w_rust = (total - pos_rust) / (pos_rust + 1e-6)
        w_scab = (total - pos_scab) / (pos_scab + 1e-6)

        return torch.tensor([w_rust, w_scab], device=self.device)

    def reconstruct_probs(self, r, s):
        """
        Reconstructs 4-class probabilities from Rust (r) and Scab (s) scores.
        Mapping:
            Healthy: (1-r)(1-s)
            Multiple: r*s
            Rust: r(1-s)
            Scab: (1-r)s
        """
        healthy = (1 - r) * (1 - s)
        multiple = r * s
        rust_only = r * (1 - s)
        scab_only = (1 - r) * s

        if isinstance(r, torch.Tensor):
            return torch.stack([healthy, multiple, rust_only, scab_only], dim=1)
        else:
            return np.stack([healthy, multiple, rust_only, scab_only], axis=1)

    def validate(self, model, loader, criterion):
        """
        Runs validation loop and calculates ROC AUC on reconstructed 4-class targets.
        """
        model.eval()
        val_loss_sum = 0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)

                val_loss_sum += loss.item()

                # Get probabilities
                probs = torch.sigmoid(logits)
                p_r = probs[:, 0]
                p_s = probs[:, 1]

                # Reconstruct predictions
                recon_probs = self.reconstruct_probs(p_r, p_s)
                all_probs.append(recon_probs.cpu())

                # Reconstruct Ground Truth
                # labels are [rust_indicator, scab_indicator]
                # If multiple: r=1, s=1
                t_r = labels[:, 0]
                t_s = labels[:, 1]

                recon_targets = torch.stack(
                    [
                        (1 - t_r) * (1 - t_s),  # Healthy
                        t_r * t_s,  # Multiple
                        t_r * (1 - t_s),  # Rust Only
                        (1 - t_r) * t_s,  # Scab Only
                    ],
                    dim=1,
                )
                all_targets.append(recon_targets.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        auc = calculate_metric(all_targets, all_probs)
        return auc, val_loss_sum / len(loader)

    def train_one_fold(self, fold, train_df, val_df, model_conf):
        print(f"\n--- Training Fold {fold} : {model_conf['name']} ---")

        # Prepare DataLoaders
        train_loader, val_loader = get_loaders(
            train_df,
            val_df,
            img_size=model_conf["img_size"],
            batch_size=model_conf["batch_size"],
        )

        # Initialize Model
        model = DiseaseClassifier(model_name=model_conf["name"], pretrained=True).to(
            self.device
        )

        # Loss Function with Class Weights
        pos_weights = None
        if Config.USE_CLASS_WEIGHTS:
            pos_weights = self.get_class_weights(train_df)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        # SWA Setup
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

        # Standard Scheduler (Cosine Annealing)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS * len(train_loader), eta_min=Config.MIN_LR
        )

        scaler = GradScaler()

        # Paths
        best_model_path = os.path.join(
            Config.WORKING_DIR, f"best_model_{model_conf['name']}_fold_{fold}.pth"
        )
        swa_model_path = os.path.join(
            Config.WORKING_DIR, f"swa_model_{model_conf['name']}_fold_{fold}.pth"
        )

        best_val_auc = 0.0

        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss_sum = 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += loss.item()

                # Step standard scheduler only before SWA starts
                if epoch < Config.SWA_START_EPOCH:
                    scheduler.step()

            avg_train_loss = train_loss_sum / len(train_loader)

            # SWA Update
            if epoch >= Config.SWA_START_EPOCH:
                swa_model.update_parameters(model)
                swa_scheduler.step()

            # Validation (Standard Model)
            val_auc, val_loss = self.validate(model, val_loader, criterion)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.6f}"
            )

            # Save Best Standard Model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        # Finalize SWA
        print("Finalizing SWA statistics...")
        update_bn(train_loader, swa_model, device=self.device)

        # Validate SWA Model
        swa_auc, swa_loss = self.validate(swa_model, val_loader, criterion)
        print(f"SWA Final AUC: {swa_auc:.6f}")

        # Save SWA Model
        torch.save(swa_model.state_dict(), swa_model_path)

    def predict_and_submit(self, test_df):
        print("\n--- Generating Submission with Rank-Calibrated Ensemble ---")

        model_preds = []

        # Iterate over all trained models (Architectures x Folds)
        for model_conf in Config.MODEL_CONFIGS:
            name = model_conf["name"]
            img_size = model_conf["img_size"]
            batch_size = model_conf["batch_size"]

            test_loader = get_test_loader(test_df, img_size, batch_size)

            for fold in range(Config.N_FOLDS):
                # We use the SWA models for inference
                model_path = os.path.join(
                    Config.WORKING_DIR, f"swa_model_{name}_fold_{fold}.pth"
                )

                if not os.path.exists(model_path):
                    print(f"Warning: Model file {model_path} not found. Skipping.")
                    continue

                # Load Model
                model = DiseaseClassifier(model_name=name, pretrained=False)
                model.load_weights(model_path, device=self.device)
                model.to(self.device)
                model.eval()

                fold_probs = []
                with torch.no_grad():
                    for imgs, _ in test_loader:
                        imgs = imgs.to(self.device)

                        # TTA: Original
                        with autocast():
                            logits = model(imgs)
                            probs = torch.sigmoid(logits)

                        # TTA: Horizontal Flip
                        imgs_flip = torch.flip(imgs, [3])
                        with autocast():
                            logits_flip = model(imgs_flip)
                            probs_flip = torch.sigmoid(logits_flip)

                        # Average TTA
                        avg_probs = (probs + probs_flip) / 2.0
                        fold_probs.append(avg_probs.cpu().numpy())

                fold_probs = np.concatenate(fold_probs, axis=0)  # Shape (N, 2)
                model_preds.append(fold_probs)

        if not model_preds:
            raise RuntimeError("No predictions generated. Check model training.")

        # Rank Normalization
        # Convert probabilities to ranks [0, 1] for each model
        ranked_preds = []
        for pred in model_preds:
            ranked_preds.append(rank_normalize(pred))

        # Average Ranks across all models
        avg_ranks = np.mean(ranked_preds, axis=0)  # Shape (N, 2)

        # Reconstruct 4-class probabilities from averaged ranks
        r_rank = avg_ranks[:, 0]
        s_rank = avg_ranks[:, 1]
        final_probs = self.reconstruct_probs(r_rank, s_rank)  # Shape (N, 4)

        # Create Submission DataFrame
        sub_df = pd.DataFrame(
            {
                "image_id": test_df["image_id"],
                "healthy": final_probs[:, 0],
                "multiple_diseases": final_probs[:, 1],
                "rust": final_probs[:, 2],
                "scab": final_probs[:, 3],
            }
        )

        # Save submission
        sub_path = "submission.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

        # Backup
        sub_df.to_csv(os.path.join(Config.WORKING_DIR, "submission.csv"), index=False)

    def run(self):
        # 1. Load Data
        train_df, val_df, test_df = load_data()

        # Combine train and val for proper 5-Fold CV
        full_df = pd.concat([train_df, val_df]).reset_index(drop=True)

        # 2. Cross Validation
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        for model_conf in Config.MODEL_CONFIGS:
            # Split based on stratify_label
            for fold, (train_idx, val_idx) in enumerate(
                skf.split(full_df, full_df["stratify_label"])
            ):
                train_sub = full_df.iloc[train_idx].reset_index(drop=True)
                val_sub = full_df.iloc[val_idx].reset_index(drop=True)

                self.train_one_fold(fold, train_sub, val_sub, model_conf)

        # 3. Inference and Submission
        self.predict_and_submit(test_df)
