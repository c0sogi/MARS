import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW

from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.dataset import AuthorDataset, create_folds, get_test_dataset

# --------------------------------------------------------------------------
# Model Components
# --------------------------------------------------------------------------


class WeightedLayerPooling(nn.Module):
    """
    Computes a learnable weighted average of the [CLS] tokens from the last N hidden layers.
    """

    def __init__(self, num_hidden_layers=4, hidden_size=1024):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.weights = nn.Parameter(torch.randn(num_hidden_layers))

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors. We take the last N.
        # Each tensor has shape (batch_size, seq_len, hidden_size)
        states = all_hidden_states[-self.num_hidden_layers :]

        # Extract [CLS] token (index 0) from each layer
        # Stack shape: (num_layers, batch_size, hidden_size)
        cls_outputs = torch.stack([s[:, 0, :] for s in states])

        # Compute softmax weights
        weights = torch.softmax(self.weights, dim=0)

        # Weighted sum: (num_layers, 1, 1) * (num_layers, batch, hidden) -> sum over layers
        weighted_output = (weights.view(-1, 1, 1) * cls_outputs).sum(dim=0)

        return weighted_output


class DebertaClassifier(nn.Module):
    """
    DeBERTa-v3-Large with Weighted Layer Pooling and a Classification Head.
    """

    def __init__(self, model_name, num_classes=3, num_pooling_layers=4):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Ensure we don't request more pooling layers than the model has
        if hasattr(self.config, "num_hidden_layers"):
            num_pooling_layers = min(num_pooling_layers, self.config.num_hidden_layers)

        self.pooling = WeightedLayerPooling(
            num_hidden_layers=num_pooling_layers, hidden_size=self.config.hidden_size
        )

        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

        # Initialize weights of the head
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.hidden_states is a tuple of (embeddings + layer_outputs)
        # We pass this to pooling
        pooled_output = self.pooling(outputs.hidden_states)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


# --------------------------------------------------------------------------
# Training Logic
# --------------------------------------------------------------------------


class TransformerTrainer:
    def __init__(self, device=Config.DEVICE):
        self.device = device

    def get_optimizer_params(self, model, learning_rate, weight_decay, layer_decay=0.9):
        """
        Implements Layer-Wise Learning Rate Decay (LLRD).
        """
        model_parameters = list(model.named_parameters())
        grouped_parameters = []

        # Identify layers
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        # Initialize groups
        # 1. Classifier / Pooling (Head) - Base LR
        # 2. Transformer Layers - Decaying LR
        # 3. Embeddings - Lowest LR

        # We assume backbone is named 'backbone'
        # DeBERTa layers are usually named 'backbone.encoder.layer.X'

        # Get number of layers from config
        num_layers = model.config.num_hidden_layers

        # Assign LR per layer
        # Layer ID N-1 (top) gets lr * (decay^0)
        # Layer ID 0 (bottom) gets lr * (decay^(N-1))
        # Embeddings get lr * (decay^N)

        for name, params in model_parameters:
            if not params.requires_grad:
                continue

            lr = learning_rate

            if "backbone.embeddings" in name:
                lr = learning_rate * (layer_decay**num_layers)
            elif "backbone.encoder.layer" in name:
                # Extract layer index
                # Format: backbone.encoder.layer.15.output...
                try:
                    layer_idx = int(
                        name.split("backbone.encoder.layer.")[1].split(".")[0]
                    )
                    # Calculate decay factor based on distance from top
                    decay_power = num_layers - 1 - layer_idx
                    lr = learning_rate * (layer_decay**decay_power)
                except:
                    lr = learning_rate  # Fallback

            # Apply weight decay logic
            if any(nd in name for nd in no_decay):
                grouped_parameters.append(
                    {"params": [params], "weight_decay": 0.0, "lr": lr}
                )
            else:
                grouped_parameters.append(
                    {"params": [params], "weight_decay": weight_decay, "lr": lr}
                )

        return grouped_parameters

    def train_one_epoch(
        self, model, dataloader, optimizer, scheduler, scaler, accumulation_steps
    ):
        model.train()
        total_loss = 0

        # No progress bar as per requirements
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(input_ids, attention_mask, labels=labels)
                loss = outputs["loss"]
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * accumulation_steps

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, model, dataloader):
        model.eval()
        preds = []
        labels_list = []
        val_loss = 0

        loss_fct = nn.CrossEntropyLoss()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(input_ids, attention_mask)
                logits = outputs["logits"]
                loss = loss_fct(logits, labels)

            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds.append(probs)
            labels_list.append(labels.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        return val_loss / len(dataloader), preds, labels_list

    def train_fold(self, fold, df_train, df_val, debug=False):
        print(f"\n[Expert A] Training Fold {fold}...")

        # Prepare Data
        train_dataset = AuthorDataset(
            texts=df_train["text"].values, labels=df_train["author"].values
        )
        val_dataset = AuthorDataset(
            texts=df_val["text"].values, labels=df_val["author"].values
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

        # Initialize Model
        model = DebertaClassifier(Config.MODEL_NAME).to(self.device)

        # Optimizer & Scheduler
        optimizer_grouped_parameters = self.get_optimizer_params(
            model, Config.LEARNING_RATE, Config.WEIGHT_DECAY, Config.LAYER_WISE_LR_DECAY
        )
        optimizer = AdamW(optimizer_grouped_parameters)

        # Calculate steps
        num_update_steps_per_epoch = (
            len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS
        )
        max_train_steps = Config.EPOCHS * num_update_steps_per_epoch
        warmup_steps = int(0.1 * max_train_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_train_steps
        )

        scaler = torch.amp.GradScaler("cuda")

        # Training Loop
        best_loss = float("inf")
        best_preds = None
        patience_counter = 0

        # Checkpoint path
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"expert_a_fold_{fold}.pt"
        )

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                Config.GRADIENT_ACCUMULATION_STEPS,
            )
            val_loss, val_preds, val_labels = self.evaluate(model, val_loader)

            # Calculate Metric
            score = compute_log_loss(val_labels, val_preds)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Log Loss: {score}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_preds = val_preds
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        # Clean up
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

        return best_preds

    def predict_test(self, fold, test_dataset):
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"expert_a_fold_{fold}.pt"
        )

        model = DebertaClassifier(Config.MODEL_NAME).to(self.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()

        dataloader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        preds = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.amp.autocast("cuda", enabled=True):
                    outputs = model(input_ids, attention_mask)
                    logits = outputs["logits"]

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds.append(probs)

        preds = np.concatenate(preds, axis=0)

        del model
        torch.cuda.empty_cache()
        gc.collect()

        return preds


# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------


def run_expert_a(load_cached_data=True, debug=False):
    """
    Main entry point for Expert A (DeBERTa).
    1. Checks cache for OOF and Test predictions.
    2. If not found, runs 5-Fold CV training.
    3. Generates and saves OOF and Test predictions.
    """
    seed_everything(Config.SEED)

    # Define cache paths (adjust for debug)
    oof_cache = Config.CACHE_EXPERT_A_OOF
    test_cache = Config.CACHE_EXPERT_A_TEST

    if debug:
        oof_cache = oof_cache.replace(".npy", "_debug.npy")
        test_cache = test_cache.replace(".npy", "_debug.npy")

    # 1. Check Cache
    if load_cached_data and os.path.exists(oof_cache) and os.path.exists(test_cache):
        print("[Expert A] Loading cached predictions...")
        oof_preds = np.load(oof_cache)
        test_preds = np.load(test_cache)
        return oof_preds, test_preds

    print("[Expert A] Cache not found or reload requested. Starting training...")

    # 2. Prepare Data
    df_folds = create_folds(load_cached_data=True, debug=debug)
    test_dataset, _ = get_test_dataset(debug=debug)

    # Initialize containers
    oof_preds = np.zeros((len(df_folds), 3))
    test_preds_accum = []

    trainer = TransformerTrainer()

    # 3. K-Fold Loop
    for fold in range(Config.N_FOLDS):
        # Split Data
        df_train = df_folds[df_folds["fold"] != fold]
        df_val = df_folds[df_folds["fold"] == fold]
        val_indices = df_val.index.values

        # Train
        val_preds = trainer.train_fold(fold, df_train, df_val, debug=debug)

        # Store OOF
        oof_preds[val_indices] = val_preds

        # Predict Test (Bagging)
        fold_test_preds = trainer.predict_test(fold, test_dataset)
        test_preds_accum.append(fold_test_preds)

    # Average Test Predictions
    test_preds_avg = np.mean(test_preds_accum, axis=0)

    # 4. Save to Cache
    np.save(oof_cache, oof_preds)
    np.save(test_cache, test_preds_avg)

    print(
        f"[Expert A] OOF Log Loss: {compute_log_loss(df_folds['author'].values, oof_preds)}"
    )
    print(f"[Expert A] Predictions saved to {oof_cache} and {test_cache}")

    return oof_preds, test_preds_avg
