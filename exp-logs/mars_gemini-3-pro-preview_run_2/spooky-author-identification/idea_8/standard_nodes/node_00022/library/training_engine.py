import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, multiclass_log_loss
from library.data_factory import DataManager, AuthorDataset
from library.neural_architecture import StylometricFusionModel


class NeuralTrainer:
    """
    Manages the training, evaluation, and inference of the StylometricFusionModel
    using Stratified K-Fold Cross-Validation.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.device = torch.device(Config.DEVICE)
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def _get_data(self, load_cached_data=True):
        """
        Loads metadata and style features. Concatenates Train and Val for CV.
        Returns combined dfs, combined style features, and test data.
        """
        train_df, val_df, test_df = DataManager.load_metadata()

        # Load style features (dense, scaled)
        train_feats, val_feats, test_feats = DataManager.get_style_features(
            train_df, val_df, test_df, load_cached_data=load_cached_data
        )

        # Concatenate Train + Val for full CV
        df_full = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        feats_full = np.concatenate([train_feats, val_feats], axis=0)

        return {
            "df_full": df_full,
            "feats_full": feats_full,
            "df_test": test_df,
            "feats_test": test_feats,
        }

    def train_fn(self, model, data_loader, optimizer, scheduler, epoch):
        """
        Executes one training epoch.
        """
        model.train()
        final_loss = 0
        accumulation_steps = Config.GRADIENT_ACCUMULATION_STEPS

        # CrossEntropyLoss expects raw logits and class indices
        criterion = nn.CrossEntropyLoss()

        for step, data in enumerate(data_loader):
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            style_features = data["style_features"].to(self.device)
            targets = data["label"].to(self.device)

            outputs = model(input_ids, attention_mask, style_features)

            loss = criterion(outputs, targets)

            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            final_loss += loss.item() * accumulation_steps

        avg_loss = final_loss / len(data_loader)
        return avg_loss

    def eval_fn(self, model, data_loader):
        """
        Executes evaluation on validation set.
        Returns average loss and predicted probabilities.
        """
        model.eval()
        final_loss = 0
        preds = []
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data in data_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                style_features = data["style_features"].to(self.device)
                targets = data["label"].to(self.device)

                outputs = model(input_ids, attention_mask, style_features)
                loss = criterion(outputs, targets)
                final_loss += loss.item()

                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                preds.append(probs.cpu().numpy())

        avg_loss = final_loss / len(data_loader)
        predictions = np.concatenate(preds)
        return avg_loss, predictions

    def predict_fn(self, model, data_loader):
        """
        Generates predictions for test set.
        """
        model.eval()
        preds = []

        with torch.no_grad():
            for data in data_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                style_features = data["style_features"].to(self.device)

                outputs = model(input_ids, attention_mask, style_features)
                probs = torch.softmax(outputs, dim=1)
                preds.append(probs.cpu().numpy())

        return np.concatenate(preds)

    def run_neural_cv(self, backbone_name, load_cached_data=True):
        """
        Orchestrates Stratified K-Fold CV for a specific backbone.
        Handles caching, training, early stopping, and inference.
        """
        # Create safe filename for caching (replace / with _)
        safe_name = backbone_name.replace("/", "_")
        oof_path = os.path.join(self.working_dir, f"oof_{safe_name}.npy")
        pred_path = os.path.join(self.working_dir, f"pred_test_{safe_name}.npy")

        # Check cache
        if load_cached_data and os.path.exists(oof_path) and os.path.exists(pred_path):
            print(f"[{backbone_name}] Loading cached predictions...")
            oof_preds = np.load(oof_path)
            test_preds = np.load(pred_path)
            return oof_preds, test_preds

        print(f"[{backbone_name}] Starting {Config.N_FOLDS}-Fold CV...")

        # Load Data
        data_bundle = self._get_data(load_cached_data)
        df_full = data_bundle["df_full"]
        feats_full = data_bundle["feats_full"]
        df_test = data_bundle["df_test"]
        feats_test = data_bundle["feats_test"]
        y_full = df_full[Config.TARGET_COL].map(Config.LABEL_MAP).values

        # Initialize Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)

        # Initialize containers
        oof_preds = np.zeros((len(df_full), Config.NUM_CLASSES))
        test_preds = np.zeros((len(df_test), Config.NUM_CLASSES))

        # Test Dataset (Fixed)
        test_dataset = AuthorDataset(
            df_test, tokenizer, feats_test, max_len=Config.MAX_LEN, is_test=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, y_full)):
            print(f"[{backbone_name}] Fold {fold + 1}/{Config.N_FOLDS}")

            # Split Data
            train_sub = df_full.iloc[train_idx].reset_index(drop=True)
            val_sub = df_full.iloc[val_idx].reset_index(drop=True)

            train_feats_sub = feats_full[train_idx]
            val_feats_sub = feats_full[val_idx]

            # Datasets & Loaders
            train_dataset = AuthorDataset(
                train_sub, tokenizer, train_feats_sub, Config.MAX_LEN
            )
            val_dataset = AuthorDataset(
                val_sub, tokenizer, val_feats_sub, Config.MAX_LEN
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.TRAIN_BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.VALID_BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Model Initialization
            model = StylometricFusionModel(
                backbone_name, num_classes=Config.NUM_CLASSES
            )
            model.to(self.device)

            # Optimizer & Scheduler
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": Config.WEIGHT_DECAY,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = torch.optim.AdamW(optimizer_parameters, lr=Config.LEARNING_RATE)
            num_train_steps = int(
                len(train_sub)
                / Config.TRAIN_BATCH_SIZE
                / Config.GRADIENT_ACCUMULATION_STEPS
                * Config.EPOCHS
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
            )

            # Training Loop with Early Stopping
            best_loss = np.inf
            best_model_state = None
            patience_counter = 0

            for epoch in range(Config.EPOCHS):
                train_loss = self.train_fn(
                    model, train_loader, optimizer, scheduler, epoch
                )
                val_loss, val_probs = self.eval_fn(model, val_loader)

                print(
                    f"  Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.15f}"
                )

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= Config.PATIENCE:
                        print(f"  Early stopping triggered at epoch {epoch+1}")
                        break

            # Load best weights
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            # Generate Predictions
            # 1. OOF
            _, final_val_probs = self.eval_fn(model, val_loader)
            oof_preds[val_idx] = final_val_probs

            # 2. Test
            fold_test_probs = self.predict_fn(model, test_loader)
            test_preds += fold_test_probs / Config.N_FOLDS

            # Cleanup
            del (
                model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                train_dataset,
                val_dataset,
            )
            torch.cuda.empty_cache()
            gc.collect()

        # Calculate overall CV score
        cv_score = multiclass_log_loss(y_full, oof_preds)
        print(f"[{backbone_name}] CV LogLoss: {cv_score:.15f}")

        # Cache results
        np.save(oof_path, oof_preds)
        np.save(pred_path, test_preds)

        return oof_preds, test_preds
