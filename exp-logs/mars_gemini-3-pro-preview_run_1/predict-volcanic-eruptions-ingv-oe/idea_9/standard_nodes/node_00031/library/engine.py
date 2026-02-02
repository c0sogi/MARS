import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from library.config import Config

# Patch Config to ensure compatibility with models.py if parameter names mismatch
if not hasattr(Config, "LGBM_PARAMS") and hasattr(Config, "LGB_PARAMS"):
    Config.LGBM_PARAMS = Config.LGB_PARAMS

from library.utils import seed_everything, calc_mae, save_submission
from library.dataset import get_tabular_data, get_vision_data, VolcanoDataset
from library.models import LGBMRegressorWrapper, EfficientNet10Ch, RidgeStacker


class Engine:
    """
    Orchestrates the Parsimonious Beamformed Stacking Strategy.
    Handles data loading, alignment, cross-validation training of heterogeneous branches,
    and final meta-learner stacking.
    """

    def __init__(self):
        self.device = Config.DEVICE
        seed_everything(Config.SEED)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def load_aligned_data(self):
        """
        Loads Tabular and Vision data for both training and validation sets,
        concatenates them for full K-Fold CV, and aligns them by segment_id.
        Also loads and aligns Test data.
        """
        print("Loading and aligning datasets...")

        # ---------------------------------------------------------
        # 1. Load Training + Validation Data (Combined for CV)
        # ---------------------------------------------------------
        # Tabular Data
        X_tab_t, _ = get_tabular_data("train")
        X_tab_v, _ = get_tabular_data("val")
        X_tab_full = pd.concat([X_tab_t, X_tab_v])

        # Vision Data
        X_vis_t, _, ids_vis_t = get_vision_data("train")
        X_vis_v, _, ids_vis_v = get_vision_data("val")
        X_vis_full = np.concatenate([X_vis_t, X_vis_v])
        ids_vis_full = np.concatenate([ids_vis_t, ids_vis_v])

        # Load Metadata to reconstruct targets reliably
        meta_t = pd.read_csv(Config.TRAIN_METADATA_PATH)
        meta_v = pd.read_csv(Config.VAL_METADATA_PATH)
        meta_full = pd.concat([meta_t, meta_v])

        # Map segment_id to target
        target_map = dict(zip(meta_full.segment_id, meta_full.time_to_eruption))

        # Find Intersection of IDs (Alignment)
        # Tabular index is segment_id
        common_ids = sorted(list(set(X_tab_full.index) & set(ids_vis_full)))

        if not common_ids:
            raise ValueError(
                "No common segment IDs found between Tabular and Vision datasets."
            )

        print(f"Aligned {len(common_ids)} training samples.")

        # Filter and Sort Tabular
        X_tab_train = X_tab_full.loc[common_ids]

        # Filter and Sort Vision
        vis_id_map = {id_: i for i, id_ in enumerate(ids_vis_full)}
        vis_indices = [vis_id_map[i] for i in common_ids]
        X_vis_train = X_vis_full[vis_indices]

        # Construct Targets
        y_train = np.array([target_map[i] for i in common_ids], dtype=np.float32)
        y_train_log = np.log1p(y_train)

        # ---------------------------------------------------------
        # 2. Load Test Data
        # ---------------------------------------------------------
        X_tab_test_raw, _ = get_tabular_data("test")
        X_vis_test_raw, _, ids_vis_test_raw = get_vision_data("test")

        # Align Test Data
        common_ids_test = sorted(
            list(set(X_tab_test_raw.index) & set(ids_vis_test_raw))
        )

        X_tab_test = X_tab_test_raw.loc[common_ids_test]

        vis_test_id_map = {id_: i for i, id_ in enumerate(ids_vis_test_raw)}
        vis_test_indices = [vis_test_id_map[i] for i in common_ids_test]
        X_vis_test = X_vis_test_raw[vis_test_indices]

        return {
            "train_ids": np.array(common_ids),
            "X_tab": X_tab_train,
            "X_vis": X_vis_train,
            "y": y_train,
            "y_log": y_train_log,
            "test_ids": np.array(common_ids_test),
            "X_tab_test": X_tab_test,
            "X_vis_test": X_vis_test,
        }

    def train_tabular_branch(self, data):
        """
        Trains the LightGBM Regressor using 5-Fold CV.
        """
        print("\n--- Training Branch A: LightGBM (Tabular) ---")
        X = data["X_tab"]
        y = data["y"]
        X_test = data["X_tab_test"]

        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            model = LGBMRegressorWrapper()
            model.fit(X_tr, y_tr, X_val, y_val)

            # Predict OOF
            val_pred = model.predict(X_val)
            oof_preds[val_idx] = val_pred

            # Predict Test
            test_preds += model.predict(X_test) / Config.N_FOLDS

            score = calc_mae(y_val, val_pred)
            scores.append(score)
            print(f"LGBM Fold {fold} MAE: {score}")

        avg_mae = np.mean(scores)
        print(f"LGBM Average MAE: {avg_mae}")
        return oof_preds, test_preds

    def train_vision_branch(self, data):
        """
        Trains the EfficientNet-B0 using 5-Fold CV with AdamW and Cosine Annealing.
        """
        print("\n--- Training Branch B: EfficientNet (Vision) ---")
        X = data["X_vis"]
        y_log = data["y_log"]
        X_test = data["X_vis_test"]

        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

        oof_preds_log = np.zeros(len(X))
        test_preds_log = np.zeros(len(X_test))
        scores = []

        # Prepare Test Loader
        test_ds = VolcanoDataset(X_test, y=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.CNN_TRAIN_PARAMS["batch_size"],
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
            print(f"Training CNN Fold {fold}...")

            # Split Data
            X_tr, y_tr = X[train_idx], y_log[train_idx]
            X_val, y_val = X[val_idx], y_log[val_idx]

            # Create Datasets & Loaders
            train_ds = VolcanoDataset(X_tr, y_tr)
            val_ds = VolcanoDataset(X_val, y_val)

            train_loader = DataLoader(
                train_ds,
                batch_size=Config.CNN_TRAIN_PARAMS["batch_size"],
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=Config.CNN_TRAIN_PARAMS["batch_size"],
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Model Setup
            model = EfficientNet10Ch().to(self.device)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.CNN_TRAIN_PARAMS["learning_rate"],
                weight_decay=Config.CNN_TRAIN_PARAMS["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=Config.CNN_TRAIN_PARAMS["epochs"],
                eta_min=Config.CNN_TRAIN_PARAMS["eta_min"],
            )
            criterion = nn.L1Loss()  # MAE on log scale

            # Training Loop
            best_val_loss = float("inf")
            best_state = None
            patience = Config.CNN_TRAIN_PARAMS["patience"]
            patience_counter = 0

            for epoch in range(Config.CNN_TRAIN_PARAMS["epochs"]):
                model.train()
                for imgs, targets in train_loader:
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device).unsqueeze(1)  # (B, 1)

                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    # Cite solution_lesson_node_00030: Stabilize training with gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for imgs, targets in val_loader:
                        imgs = imgs.to(self.device)
                        targets = targets.to(self.device).unsqueeze(1)
                        outputs = model(imgs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * imgs.size(0)

                val_loss /= len(val_ds)
                scheduler.step()

                # Checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            # Load Best Model
            if best_state is not None:
                model.load_state_dict(best_state)

            # Generate OOF Predictions
            model.eval()
            fold_oof = []
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(self.device)
                    outputs = model(imgs)
                    fold_oof.append(outputs.cpu().numpy())
            oof_preds_log[val_idx] = np.concatenate(fold_oof).flatten()

            # Generate Test Predictions
            fold_test = []
            with torch.no_grad():
                for imgs in test_loader:
                    imgs = imgs.to(self.device)
                    outputs = model(imgs)
                    fold_test.append(outputs.cpu().numpy())
            test_preds_log += np.concatenate(fold_test).flatten() / Config.N_FOLDS

            # Calculate Fold MAE (Inverse Transform)
            real_mae = calc_mae(np.expm1(y_val), np.expm1(oof_preds_log[val_idx]))
            scores.append(real_mae)
            print(f"CNN Fold {fold} MAE (Real Scale): {real_mae}")

        avg_mae = np.mean(scores)
        print(f"CNN Average MAE: {avg_mae}")

        # Return inverse transformed predictions
        return np.expm1(oof_preds_log), np.expm1(test_preds_log)

    def run(self):
        """
        Executes the full pipeline: Data Loading -> Branch A -> Branch B -> Stacking -> Submission.
        """
        # 1. Load Data
        data = self.load_aligned_data()

        # 2. Train Branch A (Tabular)
        oof_tab, test_tab = self.train_tabular_branch(data)

        # 3. Train Branch B (Vision)
        oof_vis, test_vis = self.train_vision_branch(data)

        # 4. Meta-Learner Stacking
        print("\n--- Training Meta-Learner (Ridge Stacking) ---")
        X_stack = np.column_stack([oof_tab, oof_vis])
        X_test_stack = np.column_stack([test_tab, test_vis])
        y = data["y"]

        stacker = RidgeStacker()
        stacker.fit(X_stack, y)

        final_oof = stacker.predict(X_stack)
        final_mae = calc_mae(y, final_oof)
        print(f"Final Ensemble CV MAE: {final_mae}")

        # 5. Submission
        print("Generating final submission...")
        final_preds = stacker.predict(X_test_stack)
        save_submission(data["test_ids"], final_preds)
