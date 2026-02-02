import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.dataset import AuthorDataset, load_text_data
from library.models import CustomTransformer


class AWP:
    """
    Adversarial Weight Perturbation.
    Perturbs model weights in the direction of the gradient to maximize loss,
    encouraging the model to find flatter minima.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1e-4,
        adv_eps=1e-3,
        start_epoch=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs, labels, criterion, epoch):
        """
        Performs the AWP attack and backward pass.
        1. Perturbs weights.
        2. Computes loss with perturbed weights.
        3. Backpropagates.
        4. Restores weights.
        """
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        self._attack_step()

        # Forward pass with perturbed weights
        outputs = self.model(inputs["input_ids"], inputs["attention_mask"])
        adv_loss = criterion(outputs, labels)

        # Clear gradients before backward?
        # Standard AWP usually accumulates gradients (clean + adv) or replaces them.
        # Here we assume the optimizer step will use the accumulated gradient.
        # However, PyTorch accumulates by default.
        # To strictly follow "minimize loss(w) + loss(w+delta)", we just backward again.
        adv_loss.backward()

        self._restore()
        return adv_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class NeuralTrainer:
    def __init__(self, model_name, device=Config.DEVICE, n_folds=Config.N_FOLDS):
        self.model_name = model_name
        self.device = device
        self.n_folds = n_folds
        self.num_classes = Config.NUM_CLASSES

        # Determine if we have an MLM-pretrained version available
        sanitized_name = model_name.replace("/", "-")
        mlm_path = os.path.join(Config.MLM_MODEL_DIR, f"mlm_{sanitized_name}")

        if os.path.exists(mlm_path) and os.path.exists(
            os.path.join(mlm_path, "config.json")
        ):
            print(
                f"[{model_name}] Found MLM-adapted weights at {mlm_path}. Using them."
            )
            self.load_path = mlm_path
        else:
            print(
                f"[{model_name}] No MLM weights found. Using base HuggingFace weights."
            )
            self.load_path = model_name

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.load_path)

    def _get_optimizer_scheduler(self, model, num_train_steps):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
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

        optimizer = AdamW(optimizer_parameters, lr=Config.LEARNING_RATE, eps=1e-6)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * 0.1),
            num_training_steps=num_train_steps,
        )
        return optimizer, scheduler

    def train_one_epoch(
        self, model, dataloader, optimizer, scheduler, criterion, epoch, awp=None
    ):
        model.train()
        running_loss = 0.0
        dataset_size = 0

        for step, data in enumerate(dataloader):
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            labels = data["label"].to(self.device)

            batch_size = input_ids.size(0)

            # Forward
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()

            # AWP Attack (if enabled)
            if awp is not None:
                awp.attack_backward(
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    labels,
                    criterion,
                    epoch,
                )

            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        return running_loss / dataset_size

    @torch.no_grad()
    def validate(self, model, dataloader, criterion):
        model.eval()
        running_loss = 0.0
        dataset_size = 0
        preds = []
        true_labels = []

        for data in dataloader:
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            labels = data["label"].to(self.device)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            preds.append(probs.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

        epoch_loss = running_loss / dataset_size
        preds = np.concatenate(preds, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)

        # Calculate metric using library function
        log_loss_score = calculate_log_loss(true_labels, preds)

        return epoch_loss, log_loss_score, preds

    def run_cv(self, load_cached_data=True, debug=Config.DEBUG):
        """
        Runs Stratified K-Fold Cross Validation.
        """
        seed_everything(Config.SEED)

        # 1. Load Data
        train_texts, train_labels, val_texts, val_labels, test_texts, test_ids = (
            load_text_data(load_cached_data=load_cached_data, debug=debug)
        )

        # Combine Train and Val for StratifiedKFold
        all_texts = np.concatenate([train_texts, val_texts])
        all_labels = np.concatenate([train_labels, val_labels])

        # 2. Prepare Test Dataset (Common across folds)
        test_dataset = AuthorDataset(test_texts, self.tokenizer, Config.MAX_LEN)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 3. Setup CV
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=Config.SEED
        )

        oof_preds = np.zeros((len(all_texts), self.num_classes))
        test_preds_sum = np.zeros((len(test_texts), self.num_classes))

        print(f"Starting Training for {self.model_name}...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels)):
            print(f"\n=== Fold {fold + 1}/{self.n_folds} ===")

            # Split Data
            X_train, y_train = all_texts[train_idx], all_labels[train_idx]
            X_val, y_val = all_texts[val_idx], all_labels[val_idx]

            # Create Datasets
            train_dataset = AuthorDataset(
                X_train, self.tokenizer, Config.MAX_LEN, y_train
            )
            val_dataset = AuthorDataset(X_val, self.tokenizer, Config.MAX_LEN, y_val)

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
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Initialize Model
            model = CustomTransformer(
                self.load_path, num_classes=self.num_classes, pretrained=True
            )
            model.to(self.device)

            # Optimizer & Scheduler
            num_train_steps = len(train_loader) * Config.EPOCHS
            optimizer, scheduler = self._get_optimizer_scheduler(model, num_train_steps)

            # Loss Function
            criterion = nn.CrossEntropyLoss()

            # AWP
            awp = AWP(model, optimizer, adv_lr=1e-4, adv_eps=1e-3, start_epoch=1)

            # Training Loop
            best_loss = float("inf")
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0

            for epoch in range(Config.EPOCHS):
                train_loss = self.train_one_epoch(
                    model, train_loader, optimizer, scheduler, criterion, epoch, awp
                )
                val_loss, val_log_loss, val_preds = self.validate(
                    model, val_loader, criterion
                )

                print(
                    f"Epoch {epoch+1}/{Config.EPOCHS} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Val LogLoss: {val_log_loss:.10f}"
                )

                if val_log_loss < best_loss:
                    best_loss = val_log_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    # Save OOF for best epoch
                    oof_preds[val_idx] = val_preds
                else:
                    patience_counter += 1

                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

            # Save Best Model for this Fold
            sanitized_name = self.model_name.replace("/", "-")
            save_path = os.path.join(
                Config.FINETUNED_MODEL_DIR,
                f"best_model_{sanitized_name}_fold_{fold}.pt",
            )
            torch.save(best_model_wts, save_path)

            # Load best weights for inference
            model.load_state_dict(best_model_wts)

            # Predict on Test
            _, _, fold_test_preds = self.validate(model, test_loader, criterion)
            test_preds_sum += fold_test_preds

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

        # Average Test Predictions
        test_preds_avg = test_preds_sum / self.n_folds

        overall_loss = calculate_log_loss(all_labels, oof_preds)
        print(f"\n[{self.model_name}] Overall CV Log Loss: {overall_loss:.10f}")

        return oof_preds, test_preds_avg, all_labels, test_ids

    def train_on_augmented(
        self, train_texts, train_labels, val_texts, val_labels, test_texts, test_ids
    ):
        """
        Trains the model on an augmented dataset (Stage 2) and evaluates on the original validation set.
        Does not perform K-Fold, but a single run per model instance (or can be used within a loop externally).
        For this implementation, we assume this is called per fold or as a single large run.
        Given the prompt description, we will treat this as a single training run (e.g., Fold 0 logic)
        but evaluating on the fixed val set.
        """
        seed_everything(Config.SEED)

        print(f"Starting Augmented Training for {self.model_name}...")

        # Datasets
        train_dataset = AuthorDataset(
            train_texts, self.tokenizer, Config.MAX_LEN, train_labels
        )
        val_dataset = AuthorDataset(
            val_texts, self.tokenizer, Config.MAX_LEN, val_labels
        )
        test_dataset = AuthorDataset(test_texts, self.tokenizer, Config.MAX_LEN)

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
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = CustomTransformer(
            self.load_path, num_classes=self.num_classes, pretrained=True
        )
        model.to(self.device)

        # Optimizer & Scheduler
        num_train_steps = len(train_loader) * Config.EPOCHS
        optimizer, scheduler = self._get_optimizer_scheduler(model, num_train_steps)

        # Loss Function
        criterion = nn.CrossEntropyLoss()

        # AWP
        awp = AWP(model, optimizer, adv_lr=1e-4, adv_eps=1e-3, start_epoch=1)

        best_loss = float("inf")
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion, epoch, awp
            )
            val_loss, val_log_loss, _ = self.validate(model, val_loader, criterion)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val LogLoss: {val_log_loss:.10f}"
            )

            if val_log_loss < best_loss:
                best_loss = val_log_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        # Load best weights
        model.load_state_dict(best_model_wts)

        # Predict on Test
        _, _, test_preds = self.validate(model, test_loader, criterion)

        # Predict on Val (for ensemble weights)
        _, _, val_preds = self.validate(model, val_loader, criterion)

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

        return val_preds, test_preds, best_loss
