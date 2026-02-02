import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from library.config import Config
from library.utils import compute_qwk, seed_everything
from library.dataset import get_essay_dataset


# =========================================================================================
# Pooling & Head
# =========================================================================================
class MeanMaxHead(nn.Module):
    """
    Concatenates Mean and Max pooling of the last hidden state.
    This captures both the global sentiment/argumentation (Mean) and
    salient specific features/keywords (Max).
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch, seq_len, hidden)
        # attention_mask: (batch, seq_len)

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Mean Pooling
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # Max Pooling
        # Replace padding with very small number to be ignored by max
        last_hidden_state_masked = last_hidden_state.clone()
        last_hidden_state_masked[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(last_hidden_state_masked, 1)[0]

        # Concatenate
        concat_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)

        # Projection
        output = self.fc(concat_embeddings)
        return output


# =========================================================================================
# Model Architecture
# =========================================================================================
class DebertaRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Enable gradient checkpointing for memory efficiency with large batch/seq_len
        self.backbone.gradient_checkpointing_enable()

        self.head = MeanMaxHead(self.config.hidden_size)
        self._init_weights(self.head.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        logits = self.head(last_hidden_state, attention_mask)
        return logits.squeeze(-1)


# =========================================================================================
# Adversarial Weight Perturbation (AWP)
# =========================================================================================
class AWP:
    def __init__(self, model, optimizer, adv_lr, adv_eps, start_epoch, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, input_ids, attention_mask, labels, criterion, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return

        self._save()
        self._attack_step()

        # Forward pass with perturbed weights
        # Note: We need to zero grads or handle accumulation carefully.
        # Standard AWP: Clean Grad -> Perturb -> Adv Grad -> Update
        self.optimizer.zero_grad()
        with autocast(enabled=True):
            output = self.model(input_ids, attention_mask)
            adv_loss = criterion(output, labels)

        if self.scaler:
            self.scaler.scale(adv_loss).backward()
        else:
            adv_loss.backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and param.grad.norm() > 0:
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
            if param.requires_grad and param.grad is not None and param.grad.norm() > 0:
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


# =========================================================================================
# Training Functions
# =========================================================================================
def train_one_epoch(
    model, optimizer, scheduler, dataloader, device, epoch, criterion, awp, scaler
):
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Use tqdm for progress tracking
    bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")

    for step, data in bar:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        batch_size = input_ids.size(0)

        with autocast(enabled=True):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss = loss / Config.GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
            if Config.USE_AWP:
                # Unscale gradients before AWP to get correct norms
                scaler.unscale_(optimizer)
                # AWP Attack (computes adv gradients)
                awp.attack_backward(input_ids, attention_mask, labels, criterion, epoch)

            # Clip gradients
            scaler.unscale_(optimizer)  # Safe to call again if already unscaled
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * Config.GRAD_ACCUM_STEPS * batch_size
        dataset_size += batch_size

        bar.set_postfix(loss=running_loss / dataset_size)

    return running_loss / dataset_size


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    preds = []
    targets = []

    for data in tqdm(dataloader, desc="Validating"):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        batch_size = input_ids.size(0)

        with autocast(enabled=True):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        preds.append(outputs.cpu().numpy())
        targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Compute QWK for monitoring
    # Round predictions for QWK calculation
    preds_rounded = np.rint(preds).clip(1, 6).astype(int)
    targets_int = targets.astype(int)
    qwk = compute_qwk(targets_int, preds_rounded)

    return running_loss / dataset_size, qwk, preds


def train_semantic_branch():
    """
    Main function to train the Semantic Branch (DeBERTa) on all folds.
    Saves models and generates OOF/Test predictions for stacking.
    """
    seed_everything(Config.SEED)

    # --- Data Loading ---
    print("Loading datasets for Cross-Validation...")
    # We load both train and val metadata to perform our own K-Fold split
    # consistent with Config.N_FOLDS
    train_ds_full = get_essay_dataset("train", load_cached_data=True)
    val_ds_full = get_essay_dataset("val", load_cached_data=True)

    # Combine inputs for CV splitting
    all_input_ids = np.concatenate(
        [train_ds_full.input_ids, val_ds_full.input_ids], axis=0
    )
    all_attention_mask = np.concatenate(
        [train_ds_full.attention_mask, val_ds_full.attention_mask], axis=0
    )
    all_labels = np.concatenate([train_ds_full.labels, val_ds_full.labels], axis=0)

    # Stratified KFold
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Stratify based on integer scores
    stratify_labels = all_labels.astype(int)

    oof_preds = np.zeros(len(all_labels))

    # Test Dataset (for inference after each fold)
    test_ds = get_essay_dataset("test", load_cached_data=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_preds = []

    # --- Training Loop ---
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(all_input_ids, stratify_labels)
    ):
        print(f"\n{'='*30}\nTraining Fold {fold}\n{'='*30}")

        # Create Fold Datasets
        # Helper class to wrap TensorDataset and return dict
        class FoldDataset(Dataset):
            def __init__(self, ids, mask, lbl):
                self.ids = ids
                self.mask = mask
                self.lbl = lbl

            def __len__(self):
                return len(self.ids)

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor(self.ids[idx], dtype=torch.long),
                    "attention_mask": torch.tensor(self.mask[idx], dtype=torch.long),
                    "labels": torch.tensor(self.lbl[idx], dtype=torch.float),
                }

        train_dataset = FoldDataset(
            all_input_ids[train_idx],
            all_attention_mask[train_idx],
            all_labels[train_idx],
        )
        val_dataset = FoldDataset(
            all_input_ids[val_idx], all_attention_mask[val_idx], all_labels[val_idx]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Init Model
        model = DebertaRegressor()
        model.to(Config.DEVICE)

        optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        num_train_steps = int(
            len(train_loader) / Config.GRAD_ACCUM_STEPS * Config.EPOCHS
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
            num_training_steps=num_train_steps,
        )

        # Loss & Scaler
        criterion = nn.SmoothL1Loss()
        scaler = GradScaler()

        # AWP
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.AWP_LR,
            adv_eps=Config.AWP_EPS,
            start_epoch=Config.AWP_START_EPOCH,
            scaler=scaler,
        )

        best_loss = float("inf")
        best_model_path = os.path.join(
            Config.MODEL_OUTPUT_DIR, f"deberta_fold_{fold}.bin"
        )

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model,
                optimizer,
                scheduler,
                train_loader,
                Config.DEVICE,
                epoch,
                criterion,
                awp,
                scaler,
            )

            val_loss, val_qwk, val_preds_arr = valid_one_epoch(
                model, val_loader, Config.DEVICE, criterion
            )

            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val QWK: {val_qwk:.6f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved Best Model (Loss: {best_loss:.6f})")

                # Update OOF with best model's predictions
                oof_preds[val_idx] = val_preds_arr

        # --- Inference on Test with Best Model ---
        print(f"Generating Test Predictions for Fold {fold}...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"Test Preds Fold {fold}"):
                input_ids = data["input_ids"].to(Config.DEVICE)
                attention_mask = data["attention_mask"].to(Config.DEVICE)
                with autocast(enabled=True):
                    outputs = model(input_ids, attention_mask)
                fold_test_preds.append(outputs.cpu().numpy())

        test_preds.append(np.concatenate(fold_test_preds))

        # Cleanup to free memory
        del (
            model,
            optimizer,
            scheduler,
            scaler,
            awp,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # --- Save Results ---
    avg_test_preds = np.mean(test_preds, axis=0)

    # Save OOF
    oof_df = pd.DataFrame({"score": all_labels, "pred": oof_preds})
    oof_path = os.path.join(Config.WORKING_DIR, "train_semantic_preds.parquet")
    oof_df.to_parquet(oof_path)
    print(f"Saved OOF predictions to {oof_path}")

    # Save Test
    test_pred_df = pd.DataFrame({"pred": avg_test_preds})
    test_path = os.path.join(Config.WORKING_DIR, "test_semantic_preds.parquet")
    test_pred_df.to_parquet(test_path)
    print(f"Saved Test predictions to {test_path}")

    overall_qwk = compute_qwk(
        all_labels.astype(int), np.rint(oof_preds).clip(1, 6).astype(int)
    )
    print(f"Training Complete. Overall OOF QWK: {overall_qwk:.6f}")
