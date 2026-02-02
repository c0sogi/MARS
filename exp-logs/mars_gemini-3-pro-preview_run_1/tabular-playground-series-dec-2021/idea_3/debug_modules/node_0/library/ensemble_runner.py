import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_factory import DataFactory
from library.arch_resnet import TabularResNet
from library.trainer_resnet import train_one_epoch, validate, predict as nn_predict


class EnsembleRunner:
    """
    Orchestrates the Hybrid Stacking Strategy (XGBoost + ResNet) using K-Fold Cross-Validation.
    """

    def __init__(self):
        self.device = get_device()
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        seed_everything(Config.SEED)

    def _get_data(self):
        """
        Loads engineered data and prepares full X, y for K-Fold CV.
        """
        print("Loading and engineering data...")
        # Load engineered data using DataFactory (handles caching)
        train_df, val_df, test_df, test_ids = DataFactory.load_and_engineer_data(
            load_cached_data=True
        )

        # Concatenate metadata train and val to form full training set for K-Fold
        # This allows us to use 100% of labeled data for the ensemble
        print("Concatenating Train and Val sets for K-Fold...")
        full_train = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

        # Prepare X and y
        # Target is 1-7, map to 0-6 for training
        y = (full_train[Config.TARGET_COL] - 1).values.astype(np.int64)
        X = full_train.drop(columns=[Config.TARGET_COL]).values.astype(np.float32)

        # Prepare Test
        X_test = test_df.values.astype(np.float32)

        return X, y, X_test, test_ids

    def _train_xgb_fold(self, X_train, y_train, X_val, y_val, X_test):
        """
        Trains XGBoost on a single fold.
        """
        # Create DMatrices
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        params = Config.XGB_PARAMS.copy()
        fit_params = Config.XGB_FIT_PARAMS.copy()

        # Train with early stopping
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=fit_params["num_boost_round"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=fit_params["early_stopping_rounds"],
            verbose_eval=fit_params["verbose_eval"],
        )

        # Generate probabilities
        # XGBoost predict with multi:softprob returns (N, n_classes)
        val_probs = booster.predict(dval)
        test_probs = booster.predict(dtest)

        return val_probs, test_probs

    def _train_nn_fold(self, X_train, y_train, X_val, y_val, X_test):
        """
        Trains ResNet on a single fold.
        """
        # Standard Scaling (Fit on Train, Transform Val/Test)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        # Create TensorDatasets
        train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val))
        test_ds = TensorDataset(torch.from_numpy(X_test_s))

        # Create DataLoaders
        bs = Config.NN_PARAMS["batch_size"]
        nw = Config.NN_PARAMS["num_workers"]
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

        # Initialize Model
        input_dim = X_train.shape[1]
        model = TabularResNet(
            input_dim=input_dim,
            num_classes=Config.NUM_CLASSES,
            hidden_dims=Config.NN_PARAMS["hidden_dims"],
            dropout_rate=Config.NN_PARAMS["dropout_rate"],
            use_batch_norm=Config.NN_PARAMS["use_batch_norm"],
        ).to(self.device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.NN_PARAMS["learning_rate"],
            weight_decay=Config.NN_PARAMS["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3, verbose=False
        )
        criterion = nn.CrossEntropyLoss()

        # Training Loop with Early Stopping
        best_loss = float("inf")
        patience = Config.NN_PARAMS["patience"]
        counter = 0
        best_wts = None
        epochs = Config.NN_PARAMS["epochs"]

        for epoch in range(epochs):
            train_loss, _ = train_one_epoch(
                model, train_loader, optimizer, criterion, self.device
            )
            val_loss, _ = validate(model, val_loader, criterion, self.device)

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_wts = {k: v.cpu() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        # Load best model weights
        if best_wts is not None:
            model.load_state_dict(best_wts)
        model = model.to(self.device)

        # Generate probabilities
        # nn_predict applies softmax internally
        val_probs = nn_predict(model, val_loader, self.device)
        test_probs = nn_predict(model, test_loader, self.device)

        return val_probs, test_probs

    def run_kfold_stacking(self):
        """
        Main execution method:
        1. Loads data.
        2. Runs K-Fold CV (XGB + ResNet).
        3. Trains Meta-Learner.
        4. Generates Submission.
        """
        X, y, X_test, test_ids = self._get_data()

        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # Storage for OOF predictions (N_samples, N_classes)
        xgb_oof = np.zeros((len(y), Config.NUM_CLASSES), dtype=np.float32)
        nn_oof = np.zeros((len(y), Config.NUM_CLASSES), dtype=np.float32)

        # Storage for Test predictions (Accumulator)
        xgb_test_sum = np.zeros((len(X_test), Config.NUM_CLASSES), dtype=np.float32)
        nn_test_sum = np.zeros((len(X_test), Config.NUM_CLASSES), dtype=np.float32)

        print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'='*20} Fold {fold + 1} / {Config.N_FOLDS} {'='*20}")

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            # --- 1. Train XGBoost ---
            print("Training XGBoost...")
            val_p_xgb, test_p_xgb = self._train_xgb_fold(X_tr, y_tr, X_va, y_va, X_test)
            xgb_oof[val_idx] = val_p_xgb
            xgb_test_sum += test_p_xgb

            acc_xgb = accuracy_score(y_va, np.argmax(val_p_xgb, axis=1))
            print(f"XGB Fold {fold+1} Accuracy: {acc_xgb:.6f}")

            # --- 2. Train ResNet ---
            print("Training ResNet...")
            val_p_nn, test_p_nn = self._train_nn_fold(X_tr, y_tr, X_va, y_va, X_test)
            nn_oof[val_idx] = val_p_nn
            nn_test_sum += test_p_nn

            acc_nn = accuracy_score(y_va, np.argmax(val_p_nn, axis=1))
            print(f"NN Fold {fold+1} Accuracy: {acc_nn:.6f}")

            # Clean up GPU memory
            gc.collect()
            torch.cuda.empty_cache()

        # Average Test Predictions
        xgb_test_avg = xgb_test_sum / Config.N_FOLDS
        nn_test_avg = nn_test_sum / Config.N_FOLDS

        # --- 3. Meta-Learner (Stacking) ---
        print("\nTraining Meta-Learner (Logistic Regression)...")

        # Stack OOF probabilities as features: [XGB_Probs, NN_Probs]
        # Shape: (N_samples, N_classes * 2)
        X_meta_train = np.hstack([xgb_oof, nn_oof])
        X_meta_test = np.hstack([xgb_test_avg, nn_test_avg])

        meta_model = LogisticRegression(**Config.META_PARAMS)
        meta_model.fit(X_meta_train, y)

        # Evaluate Meta-Learner on OOF
        meta_oof_preds = meta_model.predict(X_meta_train)
        meta_acc = accuracy_score(y, meta_oof_preds)
        print(f"Meta-Learner OOF Accuracy: {meta_acc:.6f}")

        # --- 4. Generate Submission ---
        print("Generating final predictions...")
        final_probs = meta_model.predict_proba(X_meta_test)
        # Convert 0-6 back to 1-7
        final_preds = np.argmax(final_probs, axis=1) + 1

        print(f"Saving submission to {Config.SUBMISSION_PATH}")
        submission = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: final_preds}
        )
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Ensemble execution complete.")
