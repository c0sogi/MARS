import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from library.config import Config
from library.utils import get_logger, seed_everything, compute_qwk
from library.data import load_data, get_collate_fn

# --- Pooling Layers ---


class WeightedLayerPooling(nn.Module):
    """
    Computes a weighted average of the hidden states from the last N layers.
    The weights are learnable parameters.
    """

    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(
                torch.tensor(
                    [1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float
                )
            )
        )

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of (batch, seq_len, hidden_dim) tensors
        # We select the layers from layer_start to the end
        selected_layers = all_hidden_states[self.layer_start :]

        # Stack them: (num_selected, batch, seq_len, hidden_dim)
        all_layer_embedding = torch.stack(selected_layers)

        # Softmax weights to ensure they sum to 1
        weight_factor = nn.functional.softmax(self.layer_weights, dim=0).view(
            -1, 1, 1, 1
        )

        # Weighted sum: (batch, seq_len, hidden_dim)
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0)
        return weighted_average


# --- Model Architecture ---


class CustomDeberta(nn.Module):
    """
    DeBERTa-v3-Large with Weighted Layer Pooling and Concatenated Mean/Max Pooling.
    """

    def __init__(self, model_name, config_path=None, pretrained=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # Weighted Layer Pooling for the last 4 layers
        # DeBERTa Large has 24 layers (indices 0-23). all_hidden_states includes embedding (idx 0) + 24 layers = 25 tensors.
        # We want the last 4 layers.
        # However, usually all_hidden_states has len = num_layers + 1.
        # We want indices [-4, -3, -2, -1].
        self.pooler = WeightedLayerPooling(
            num_hidden_layers=self.config.num_hidden_layers,
            layer_start=self.config.num_hidden_layers + 1 - 4,
            layer_weights=nn.Parameter(
                torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
            ),
        )

        self.fc = nn.Linear(self.config.hidden_size * 2, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def feature(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states

        # Get weighted average of last 4 layers -> (batch, seq_len, hidden_dim)
        weighted_embedding = self.pooler(all_hidden_states)

        # Mask padding tokens for pooling
        # attention_mask: (batch, seq_len) -> (batch, seq_len, 1)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(weighted_embedding.size()).float()
        )

        # Mean Pooling
        sum_embeddings = torch.sum(weighted_embedding * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # Max Pooling
        # Set padding tokens to large negative value
        weighted_embedding[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(weighted_embedding, 1)[0]

        # Concatenate
        concat_embeddings = torch.cat([mean_embeddings, max_embeddings], 1)
        return concat_embeddings

    def forward(self, input_ids, attention_mask):
        feature = self.feature(input_ids, attention_mask)
        output = self.fc(feature)
        return output


# --- Adversarial Weight Perturbation (AWP) ---


class AWP:
    """
    Adversarial Weight Perturbation.
    Perturbs weights to maximize loss, improving robustness.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, inputs, criterion, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        self._attack_step()

        # Forward pass with perturbed weights
        with torch.cuda.amp.autocast(enabled=True):
            y_preds = self.model(inputs["input_ids"], inputs["attention_mask"])
            adv_loss = criterion(y_preds.view(-1), inputs["labels"].view(-1))

        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(adv_loss).backward()
        else:
            adv_loss.backward()

        self._restore()

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


# --- Training Helper Functions ---


def train_fn(
    model, train_loader, optimizer, scheduler, criterion, epoch, device, awp=None
):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses = []

    for step, batch in enumerate(train_loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                batch[k] = [
                    x.to(device) for x in v
                ]  # Should not happen with collate_fn returning tensors

        with torch.cuda.amp.autocast(enabled=True):
            output = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(output.view(-1), batch["labels"].view(-1))

        # Normalize loss for gradient accumulation
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        # AWP Attack
        if awp is not None:
            awp.attack_backward(batch, criterion, epoch)

        if (step + 1) % Config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        losses.append(loss.item() * Config.gradient_accumulation_steps)

    return np.mean(losses)


def valid_fn(model, valid_loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    labels = []

    with torch.no_grad():
        for batch in valid_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            output = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(output.view(-1), batch["labels"].view(-1))

            losses.append(loss.item())
            preds.append(output.view(-1).cpu().numpy())
            labels.append(batch["labels"].view(-1).cpu().numpy())

    predictions = np.concatenate(preds)
    true_labels = np.concatenate(labels)

    return np.mean(losses), predictions, true_labels


def inference_fn(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            output = model(batch["input_ids"], batch["attention_mask"])
            preds.append(output.view(-1).cpu().numpy())

    return np.concatenate(preds)


# --- Main Training Pipeline ---


def run_semantic_training():
    """
    Orchestrates the training of the Deep Semantic Branch (DeBERTa).
    """
    logger = get_logger("semantic_train")
    seed_everything(Config.seed)

    os.makedirs(Config.model_dir, exist_ok=True)
    os.makedirs(Config.output_dir, exist_ok=True)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Load Data
    logger.info("Loading datasets...")
    train_dataset_full = load_data(tokenizer, split="train", debug=Config.debug)
    val_dataset_full = load_data(tokenizer, split="val", debug=Config.debug)

    # Combine for cross-validation logic
    # In this specific setup, the metadata already provides a fixed train/val split.
    # However, to perform 5-fold CV as requested in the Idea, we should merge and re-split,
    # OR if the metadata implies a single holdout, we respect that.
    # The Idea says: "Splitting: We will employ Stratified 5-Fold Cross-Validation".
    # The metadata provides 'train.csv' and 'val.csv' which is a single split (80/20).
    # To do 5-fold CV properly, we need to combine them and use StratifiedKFold.

    # 1. Merge datasets
    full_df = pd.concat([train_dataset_full.df, val_dataset_full.df]).reset_index(
        drop=True
    )

    # 2. Stratified K-Fold
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Prepare OOF array
    oof_preds = np.zeros(len(full_df))

    # Loop over folds
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_df, full_df["score"].astype(str))
    ):
        logger.info(f"--- Fold {fold} ---")

        # Create subsets
        df_train = full_df.iloc[train_idx].reset_index(drop=True)
        df_val = full_df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_ds = load_data(
            tokenizer, split="train", debug=Config.debug
        )  # Dummy call to get class
        train_ds.df = df_train
        train_ds.texts = df_train["full_text"].values
        train_ds.labels = df_train["score"].values.astype(float)

        val_ds = load_data(tokenizer, split="train", debug=Config.debug)  # Dummy call
        val_ds.df = df_val
        val_ds.texts = df_val["full_text"].values
        val_ds.labels = df_val["score"].values.astype(float)

        # DataLoaders
        collate_fn = get_collate_fn(tokenizer)
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            val_ds,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Model Setup
        model = CustomDeberta(Config.model_name, pretrained=True)
        model.to(Config.device)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )
        num_train_steps = int(
            len(train_ds)
            / Config.train_batch_size
            / Config.gradient_accumulation_steps
            * Config.epochs
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Loss
        criterion = nn.SmoothL1Loss()

        # AWP
        awp = None
        if Config.use_awp:
            awp = AWP(
                model,
                optimizer,
                adv_lr=Config.awp_lr,
                adv_eps=Config.awp_eps,
                start_epoch=Config.awp_start_epoch,
                scaler=None,  # Will be handled inside train_fn
            )

        best_qwk = -1.0
        best_loss = np.inf

        for epoch in range(Config.epochs):
            start_time = time.time()

            train_loss = train_fn(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                epoch,
                Config.device,
                awp,
            )
            val_loss, val_preds, val_labels = valid_fn(
                model, valid_loader, criterion, Config.device
            )

            # Calculate QWK
            val_qwk = compute_qwk(val_labels, val_preds)

            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val QWK: {val_qwk:.6f} - Time: {elapsed:.0f}s"
            )

            # Save best model
            if val_qwk > best_qwk:
                best_qwk = val_qwk
                best_loss = val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(Config.model_dir, f"deberta_fold_{fold}.bin"),
                )
                logger.info(f"  -> Saved Best Model (QWK: {best_qwk:.6f})")

        # Load best model for OOF
        model.load_state_dict(
            torch.load(os.path.join(Config.model_dir, f"deberta_fold_{fold}.bin"))
        )
        _, oof_val_preds, _ = valid_fn(model, valid_loader, criterion, Config.device)
        oof_preds[val_idx] = oof_val_preds

        del model, optimizer, scheduler, train_loader, valid_loader
        torch.cuda.empty_cache()
        gc.collect()

    # Save OOF predictions
    oof_path = os.path.join(Config.output_dir, "semantic_oof.npy")
    np.save(oof_path, oof_preds)
    logger.info(f"Saved OOF predictions to {oof_path}")

    # Calculate Overall CV Score
    overall_qwk = compute_qwk(full_df["score"].values, oof_preds)
    logger.info(f"Overall CV QWK: {overall_qwk:.6f}")

    return oof_preds


def predict_semantic_test():
    """
    Generates predictions for the test set using the trained models from all folds.
    """
    logger = get_logger("semantic_test")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    test_ds = load_data(tokenizer, split="test", debug=Config.debug)

    collate_fn = get_collate_fn(tokenizer)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    fold_preds = []

    for fold in range(Config.n_folds):
        model_path = os.path.join(Config.model_dir, f"deberta_fold_{fold}.bin")
        if not os.path.exists(model_path):
            logger.warning(
                f"Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        logger.info(f"Predicting with Fold {fold}...")
        model = CustomDeberta(Config.model_name, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)

        preds = inference_fn(model, test_loader, Config.device)
        fold_preds.append(preds)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if not fold_preds:
        raise RuntimeError("No models found for prediction.")

    avg_preds = np.mean(fold_preds, axis=0)

    # Save test predictions
    save_path = os.path.join(Config.output_dir, "semantic_test_preds.npy")
    np.save(save_path, avg_preds)
    logger.info(f"Saved Test predictions to {save_path}")

    return avg_preds
