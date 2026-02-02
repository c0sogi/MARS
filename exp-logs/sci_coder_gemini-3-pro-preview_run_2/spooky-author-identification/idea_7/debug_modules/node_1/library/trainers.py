import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb

from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.dataset import AuthorDataset
from library.neural_net import DebertaWithMSD


class ModelTrainer:
    """
    Encapsulates training logic for Neural and Classical models.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.device = Config.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    def train_neural_fold(
        self, train_texts, train_labels, val_texts, val_labels, fold_idx
    ):
        """
        Trains the DeBERTa model for a single fold.

        Args:
            train_texts: List/Array of training text.
            train_labels: List/Array of training labels.
            val_texts: List/Array of validation text.
            val_labels: List/Array of validation labels.
            fold_idx: Integer index of the current fold (for logging/saving).

        Returns:
            oof_preds: Numpy array of probabilities for the validation set.
            best_model_state: State dictionary of the best model.
        """
        print(f"\n[Neural Fold {fold_idx}] Starting training...")

        # 1. Prepare Datasets and Loaders
        train_dataset = AuthorDataset(
            train_texts, train_labels, self.tokenizer, max_length=Config.MAX_LENGTH
        )
        val_dataset = AuthorDataset(
            val_texts, val_labels, self.tokenizer, max_length=Config.MAX_LENGTH
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 2. Model, Optimizer, Scheduler
        model = DebertaWithMSD(num_classes=3).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        num_train_steps = int(
            len(train_loader) / Config.ACCUMULATION_STEPS * Config.EPOCHS
        )
        num_warmup_steps = int(num_train_steps * 0.1)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        # 3. Training Loop
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        oof_preds = None

        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss_accum = 0.0

            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["target"].to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, targets)
                    loss = loss / Config.ACCUMULATION_STEPS

                scaler.scale(loss).backward()
                train_loss_accum += loss.item() * Config.ACCUMULATION_STEPS

                if (step + 1) % Config.ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), Config.MAX_GRAD_NORM
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

            avg_train_loss = train_loss_accum / len(train_loader)

            # 4. Validation
            model.eval()
            val_preds_list = []
            val_targets_list = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    targets = batch["target"].to(self.device)

                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids, attention_mask)

                    # Apply softmax to get probabilities
                    probs = torch.softmax(outputs, dim=1)

                    val_preds_list.append(probs.cpu().numpy())
                    val_targets_list.append(targets.cpu().numpy())

            val_probs = np.concatenate(val_preds_list, axis=0)
            val_true = np.concatenate(val_targets_list, axis=0)

            val_loss = calculate_log_loss(val_true, val_probs)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.15f}"
            )

            # 5. Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                oof_preds = val_probs
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        if oof_preds is None:
            # Fallback if training fails or 1 epoch only (should not happen usually)
            oof_preds = val_probs
            best_model_state = model.state_dict()

        return oof_preds, best_model_state

    def train_classical_fold(
        self, X_train, y_train, X_val, y_val, model_type, fold_idx
    ):
        """
        Trains a classical machine learning model (LR, NB, XGB) for a single fold.

        Args:
            X_train: Training features (Sparse or Dense).
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            model_type: String identifier ('lr', 'nb', 'xgb').
            fold_idx: Integer index of the current fold.

        Returns:
            oof_preds: Numpy array of probabilities for the validation set.
            model: The trained model object.
        """
        print(f"\n[Classical Fold {fold_idx}] Training {model_type.upper()}...")

        model = None

        if model_type == "lr":
            # Logistic Regression for sparse features
            model = LogisticRegression(
                C=1.0,
                solver="liblinear",
                multi_class="ovr",
                random_state=Config.SEED,
                max_iter=1000,
            )
            model.fit(X_train, y_train)

        elif model_type == "nb":
            # Naive Bayes for sparse features
            model = MultinomialNB(alpha=0.02)
            model.fit(X_train, y_train)

        elif model_type == "xgb":
            # XGBoost for dense features
            model = xgb.XGBClassifier(
                n_estimators=2000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=3,
                n_jobs=-1,
                random_state=Config.SEED,
                verbosity=0,
                device="cuda" if torch.cuda.is_available() else "cpu",
                early_stopping_rounds=50,
            )

            # XGBoost supports early stopping
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Inference
        val_probs = model.predict_proba(X_val)

        # Metric
        val_loss = calculate_log_loss(y_val, val_probs)
        print(f"[{model_type.upper()} Fold {fold_idx}] Val Loss: {val_loss:.15f}")

        return val_probs, model
