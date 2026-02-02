import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import os
import gc
import time

from library.config import Config
from library.dataset import FeatureDataset, TestFeatureDataset
from library.feature_extract import FeatureExtractor

# ==========================================
# MODEL ARCHITECTURE
# ==========================================


class DualStreamMultiTaskNetwork(nn.Module):
    def __init__(self):
        super(DualStreamMultiTaskNetwork, self).__init__()

        # 1. Feature Projection Streams
        # Project disparate backbone features to a common latent dimension
        self.proj_resnet = nn.Sequential(
            nn.Linear(Config.RESNET_DIM, Config.PROJECTION_DIM),
            nn.BatchNorm1d(Config.PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        self.proj_effnet = nn.Sequential(
            nn.Linear(Config.EFFNET_DIM, Config.PROJECTION_DIM),
            nn.BatchNorm1d(Config.PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # 2. Shared Trunk
        # Input Dimension = 1024 (ResNet Proj) + 1024 (EffNet Proj) = 2048
        self.trunk = nn.Sequential(
            nn.Linear(Config.PROJECTION_DIM * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # 3. Multi-Task Heads
        self.head_l1 = nn.Linear(512, Config.NUM_CLASSES_L1)
        self.head_l2 = nn.Linear(512, Config.NUM_CLASSES_L2)
        self.head_l3 = nn.Linear(512, Config.NUM_CLASSES_L3)

    def forward(self, resnet_feat, effnet_feat):
        # Project
        p_res = self.proj_resnet(resnet_feat)
        p_eff = self.proj_effnet(effnet_feat)

        # Fuse
        combined = torch.cat([p_res, p_eff], dim=1)

        # Shared Representation
        feat = self.trunk(combined)

        # Classification
        l1_logits = self.head_l1(feat)
        l2_logits = self.head_l2(feat)
        l3_logits = self.head_l3(feat)

        return l1_logits, l2_logits, l3_logits


# ==========================================
# TRAINER CLASS
# ==========================================


class Trainer:
    def __init__(self, model, device, train_loader, val_loader):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: OneCycle for super-convergence
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=Config.EPOCHS,
            pct_start=0.3,
        )

    def train_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            # Move to device
            r_feat = batch["resnet_feat"].to(self.device)
            e_feat = batch["effnet_feat"].to(self.device)
            l1 = batch["label_l1"].to(self.device)
            l2 = batch["label_l2"].to(self.device)
            l3 = batch["label_l3"].to(self.device)

            # MixUp
            use_mixup = True
            if use_mixup:
                alpha = Config.MIXUP_ALPHA
                lam = np.random.beta(alpha, alpha)

                batch_size = r_feat.size(0)
                index = torch.randperm(batch_size).to(self.device)

                # Mix Features
                mixed_r = lam * r_feat + (1 - lam) * r_feat[index]
                mixed_e = lam * e_feat + (1 - lam) * e_feat[index]

                # Forward
                pred_l1, pred_l2, pred_l3 = self.model(mixed_r, mixed_e)

                # Loss (Mix of targets)
                loss_l1 = lam * self.criterion(pred_l1, l1) + (
                    1 - lam
                ) * self.criterion(pred_l1, l1[index])
                loss_l2 = lam * self.criterion(pred_l2, l2) + (
                    1 - lam
                ) * self.criterion(pred_l2, l2[index])
                loss_l3 = lam * self.criterion(pred_l3, l3) + (
                    1 - lam
                ) * self.criterion(pred_l3, l3[index])

            else:
                pred_l1, pred_l2, pred_l3 = self.model(r_feat, e_feat)
                loss_l1 = self.criterion(pred_l1, l1)
                loss_l2 = self.criterion(pred_l2, l2)
                loss_l3 = self.criterion(pred_l3, l3)

            total_loss = loss_l1 + loss_l2 + loss_l3

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running_loss += total_loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        correct_l3 = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                r_feat = batch["resnet_feat"].to(self.device)
                e_feat = batch["effnet_feat"].to(self.device)
                l3 = batch["label_l3"].to(self.device)

                _, _, out_l3 = self.model(r_feat, e_feat)

                preds = torch.argmax(out_l3, dim=1)
                correct_l3 += (preds == l3).sum().item()
                total += l3.size(0)

        return correct_l3 / total if total > 0 else 0.0

    def fit(self):
        best_acc = 0.0
        patience_counter = 0
        best_state = None

        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_acc = self.validate()

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {train_loss:.6f} | Val L3 Acc: {val_acc:.10f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

        # Load best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_acc


# ==========================================
# MAIN PIPELINE
# ==========================================


def run_task():
    # 1. Feature Extraction
    print("Step 1: Checking/Extracting Features...")
    extractor = FeatureExtractor(debug_size=Config.DEBUG_SIZE)
    extractor.extract_all()

    # 2. Prepare Datasets
    print("Step 2: Loading Datasets...")
    train_ds = FeatureDataset(
        Config.TRAIN_FEATS_RESNET, Config.TRAIN_FEATS_EFFNET, Config.TRAIN_LABELS
    )
    val_ds = FeatureDataset(
        Config.VAL_FEATS_RESNET, Config.VAL_FEATS_EFFNET, Config.VAL_LABELS
    )
    test_ds = TestFeatureDataset(
        Config.TEST_FEATS_RESNET, Config.TEST_FEATS_EFFNET, Config.TEST_IDS
    )

    # Handle Debug Sizing
    if Config.DEBUG_SIZE:
        indices = list(range(min(len(train_ds), Config.DEBUG_SIZE)))
        train_ds = Subset(train_ds, indices)

        v_indices = list(range(min(len(val_ds), Config.DEBUG_SIZE)))
        val_ds = Subset(val_ds, v_indices)

        t_indices = list(range(min(len(test_ds), Config.DEBUG_SIZE)))
        test_ds = Subset(test_ds, t_indices)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device(Config.DEVICE)

    # 3. Ensemble Training & Inference
    print("Step 3: Training Ensemble (3 Models)...")

    # We accumulate probabilities on CPU to save GPU memory
    # Shape: (Num_Test_Samples, Num_L3_Classes)
    num_test_samples = len(test_ds)
    accumulated_probs = torch.zeros(
        (num_test_samples, Config.NUM_CLASSES_L3), dtype=torch.float32
    )
    test_ids = []

    # Collect IDs once
    print("Collecting Test IDs...")
    for batch in test_loader:
        test_ids.extend(batch["_id"].numpy())
    test_ids = np.array(test_ids)

    # Ensemble Loop
    for i in range(3):
        print(f"\n--- Training Model {i+1}/3 ---")

        # Seed for diversity
        seed = Config.SEED + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = DualStreamMultiTaskNetwork().to(device)
        trainer = Trainer(model, device, train_loader, val_loader)
        trainer.fit()

        # Inference
        print(f"Generating predictions for Model {i+1}...")
        model.eval()
        batch_start = 0
        with torch.no_grad():
            for batch in test_loader:
                r_feat = batch["resnet_feat"].to(device)
                e_feat = batch["effnet_feat"].to(device)

                _, _, logits_l3 = model(r_feat, e_feat)
                probs = F.softmax(logits_l3, dim=1).cpu()

                batch_size = probs.size(0)
                accumulated_probs[batch_start : batch_start + batch_size] += probs
                batch_start += batch_size

        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Final Submission
    print("\nStep 4: Generating Submission...")
    final_preds = torch.argmax(accumulated_probs, dim=1).numpy()

    df_sub = pd.DataFrame({"_id": test_ids, "category_id": final_preds})

    # Map back from index to category_id using HierarchyMapper
    # We need to load the mapper to get the index->ID mapping
    from library.utils import HierarchyMapper

    mapper = HierarchyMapper(load_cached_data=True)

    # The model predicts indices 0..N-1. We need to map these back to original category_ids
    # Using a vectorized map
    idx_to_id_map = np.zeros(Config.NUM_CLASSES_L3, dtype=np.int64)
    for idx, cat_id in mapper.l3_idx_to_id.items():
        idx_to_id_map[idx] = cat_id

    df_sub["category_id"] = idx_to_id_map[df_sub["category_id"].values]

    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
