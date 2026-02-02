import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import ModelConfig, TrainingConfig, PathConfig, FeatureConfig
from library.utils import seed_everything
from library.models import TripleBranchMLP
from library.data_factory import (
    load_and_preprocess,
    get_pytorch_dataloaders,
    PizzaDataset,
)

# Set device
DEVICE = TrainingConfig.DEVICE


def train_rf(X_train, y_train, X_val=None, y_val=None):
    """
    Trains the Random Forest Base Learner (Stream A).
    """
    print("Training Random Forest...")
    rf = RandomForestClassifier(**ModelConfig.RF_PARAMS)
    rf.fit(X_train, y_train)

    # Validation score if provided
    if X_val is not None and y_val is not None:
        preds = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        print(f"Random Forest Validation AUC: {auc:.16f}")

    return rf


def train_mlp(train_loader, val_loader, meta_dim):
    """
    Trains the Triple-Branch MLP Base Learner (Stream B).
    """
    print("Training Triple-Branch MLP...")
    model = TripleBranchMLP(meta_dim=meta_dim).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=TrainingConfig.LEARNING_RATE,
        weight_decay=TrainingConfig.WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(TrainingConfig.EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            sem = batch["semantic_input"].to(DEVICE)
            comm = batch["community_input"].to(DEVICE)
            meta = batch["meta_input"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(sem, comm, meta)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sem.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                sem = batch["semantic_input"].to(DEVICE)
                comm = batch["community_input"].to(DEVICE)
                meta = batch["meta_input"].to(DEVICE)
                target = batch["target"]  # Keep on CPU for sklearn

                logits = model(sem, comm, meta)
                probs = torch.sigmoid(logits).cpu().numpy()

                val_preds.extend(probs)
                val_targets.extend(target.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            val_auc = 0.5  # Handle edge cases with single class in batch

        # print(f"Epoch {epoch+1}/{TrainingConfig.EPOCHS} - Loss: {train_loss:.6f} - Val AUC: {val_auc:.6f}")

        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= TrainingConfig.PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best AUC: {best_auc:.16f}"
            )
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_auc


def predict_mlp(model, loader):
    """
    Generates probabilities using the trained MLP.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            sem = batch["semantic_input"].to(DEVICE)
            comm = batch["community_input"].to(DEVICE)
            meta = batch["meta_input"].to(DEVICE)

            logits = model(sem, comm, meta)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)
    return np.array(preds).flatten()


def train_stacker(X_meta, y):
    """
    Trains the Level-2 Logistic Regression Stacker.
    """
    print("Training Meta-Learner (Stacker)...")
    # C=1.0 is default, but explicit for clarity based on config
    lr = LogisticRegression(
        C=ModelConfig.STACKING_LR_C, random_state=TrainingConfig.SEED, solver="lbfgs"
    )
    lr.fit(X_meta, y)
    return lr


class CrossValidator:
    def __init__(self, k_folds=TrainingConfig.NUM_FOLDS):
        self.k_folds = k_folds
        self.seed = TrainingConfig.SEED

    def _prepare_rf_data(self, data_dict):
        """Concatenates features for RF: TFIDF + Community + Meta/Num"""
        return np.hstack(
            [data_dict["tfidf"], data_dict["community"], data_dict["meta_num"]]
        )

    def _slice_dict(self, data_dict, indices):
        """Slices all arrays in the dictionary based on indices."""
        sliced = {}
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                sliced[k] = v[indices]
            else:
                sliced[k] = v  # Pass through non-array items if any
        return sliced

    def run(self):
        seed_everything(self.seed)

        # 1. Load Data
        print("Loading and Preprocessing Data...")
        # stream_a = (train, val, test), stream_b = (train, val, test)
        stream_a, stream_b = load_and_preprocess(load_cached_data=True)

        # Unpack Training Data (we perform CV on the 'train' split)
        # We will also predict on the provided 'val' and 'test' sets at each fold
        data_a_train, data_a_val_holdout, data_a_test = stream_a
        data_b_train, data_b_val_holdout, data_b_test = stream_b

        y_full = data_a_train["y"]  # Targets
        N_train = len(y_full)

        # Prepare Holdout/Test Data for RF
        X_rf_val_holdout = self._prepare_rf_data(data_a_val_holdout)
        X_rf_test = self._prepare_rf_data(data_a_test)

        # Prepare Holdout/Test Loaders for MLP
        # Note: We create datasets/loaders once here, but for training we need to split the train set
        _, loader_mlp_val_holdout, loader_mlp_test = get_pytorch_dataloaders(
            data_b_train,
            data_b_val_holdout,
            data_b_test,
            batch_size=TrainingConfig.BATCH_SIZE,
        )

        # Storage for OOF and Test Preds
        oof_preds_rf = np.zeros(N_train)
        oof_preds_mlp = np.zeros(N_train)

        test_preds_rf_accum = np.zeros(len(data_a_test["ids"]))
        test_preds_mlp_accum = np.zeros(len(data_a_test["ids"]))

        val_holdout_preds_rf_accum = np.zeros(len(data_a_val_holdout["y"]))
        val_holdout_preds_mlp_accum = np.zeros(len(data_a_val_holdout["y"]))

        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=self.k_folds, shuffle=True, random_state=self.seed
        )

        print(f"\nStarting {self.k_folds}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(N_train), y_full)
        ):
            print(f"\n=== Fold {fold + 1} ===")

            # --- Prepare Fold Data ---
            # Stream A (RF)
            X_rf_full = self._prepare_rf_data(data_a_train)
            X_rf_fold_train = X_rf_full[train_idx]
            y_fold_train = y_full[train_idx]
            X_rf_fold_val = X_rf_full[val_idx]
            y_fold_val = y_full[val_idx]

            # Stream B (MLP)
            dict_b_fold_train = self._slice_dict(data_b_train, train_idx)
            dict_b_fold_val = self._slice_dict(data_b_train, val_idx)

            loader_fold_train, loader_fold_val, _ = get_pytorch_dataloaders(
                dict_b_fold_train, dict_b_fold_val, data_b_test  # test unused here
            )

            # --- Train RF ---
            rf_model = train_rf(
                X_rf_fold_train, y_fold_train, X_rf_fold_val, y_fold_val
            )

            # Predict OOF
            oof_preds_rf[val_idx] = rf_model.predict_proba(X_rf_fold_val)[:, 1]
            # Predict Test & Holdout (Accumulate)
            test_preds_rf_accum += rf_model.predict_proba(X_rf_test)[:, 1]
            val_holdout_preds_rf_accum += rf_model.predict_proba(X_rf_val_holdout)[:, 1]

            # --- Train MLP ---
            meta_dim = dict_b_fold_train["meta_num"].shape[1]
            mlp_model, best_auc = train_mlp(
                loader_fold_train, loader_fold_val, meta_dim
            )

            # Predict OOF
            oof_preds_mlp[val_idx] = predict_mlp(mlp_model, loader_fold_val)
            # Predict Test & Holdout (Accumulate)
            test_preds_mlp_accum += predict_mlp(mlp_model, loader_mlp_test)
            val_holdout_preds_mlp_accum += predict_mlp(
                mlp_model, loader_mlp_val_holdout
            )

        # Average Predictions
        test_preds_rf_avg = test_preds_rf_accum / self.k_folds
        test_preds_mlp_avg = test_preds_mlp_accum / self.k_folds

        val_holdout_preds_rf_avg = val_holdout_preds_rf_accum / self.k_folds
        val_holdout_preds_mlp_avg = val_holdout_preds_mlp_accum / self.k_folds

        # --- Stacking ---
        print("\n=== Training Meta-Learner ===")

        # Form Meta-Features (OOF)
        X_meta_train = np.column_stack([oof_preds_rf, oof_preds_mlp])

        # Train Stacker
        stacker = train_stacker(X_meta_train, y_full)

        # Evaluate Stacker on OOF (Internal CV Score)
        oof_final_preds = stacker.predict_proba(X_meta_train)[:, 1]
        oof_auc = roc_auc_score(y_full, oof_final_preds)
        print(f"Stacker OOF AUC: {oof_auc:.16f}")
        print(
            f"Stacker Coefficients: RF={stacker.coef_[0][0]:.4f}, MLP={stacker.coef_[0][1]:.4f}"
        )

        # Evaluate Stacker on Holdout Validation Set
        X_meta_val_holdout = np.column_stack(
            [val_holdout_preds_rf_avg, val_holdout_preds_mlp_avg]
        )
        val_final_preds = stacker.predict_proba(X_meta_val_holdout)[:, 1]
        val_holdout_auc = roc_auc_score(data_a_val_holdout["y"], val_final_preds)
        print(f"Stacker Holdout Validation AUC: {val_holdout_auc:.16f}")

        # --- Submission ---
        print("\nGenerating Submission...")
        X_meta_test = np.column_stack([test_preds_rf_avg, test_preds_mlp_avg])
        final_test_probs = stacker.predict_proba(X_meta_test)[:, 1]

        self._save_submission(data_a_test["ids"], final_test_probs)

    def _save_submission(self, ids, probs):
        df_sub = pd.DataFrame({"request_id": ids, "requester_received_pizza": probs})

        # Ensure directory exists
        os.makedirs(PathConfig.SUBMISSION_DIR, exist_ok=True)

        # Save
        df_sub.to_csv(PathConfig.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {PathConfig.SUBMISSION_FILE}")
        print(df_sub.head())
