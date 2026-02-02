import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AutoConfig,
)
from tqdm import (
    tqdm,
)  # Not used for printing, but good practice to have available if needed, though we suppress output

from library.configuration import Config
from library.utilities import seed_everything, compute_log_loss
from library.data_handling import AuthorDataset, get_stratified_folds


class DebertaWithPooling(nn.Module):
    """
    DeBERTa-v3-Large with Concatenated Mean + Max Pooling Head.
    """

    def __init__(self, model_name, num_classes, dropout_prob=0.1):
        super(DebertaWithPooling, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Hidden size is doubled because of concatenation (Mean + Max)
        self.fc = nn.Linear(self.config.hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

        # Initialize weights for the new head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (Batch, Seq, Hidden)

        # Mask padding tokens
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Mean Pooling
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooling = sum_embeddings / sum_mask

        # Max Pooling
        # Set padding tokens to large negative value so they aren't selected as max
        last_hidden_state[input_mask_expanded == 0] = -1e9
        max_pooling, _ = torch.max(last_hidden_state, 1)

        # Concatenate
        concat_vector = torch.cat([mean_pooling, max_pooling], 1)

        # Classification Head
        x = self.dropout(concat_vector)
        logits = self.fc(x)

        return logits


def get_llrd_optimizer(model, learning_rate, weight_decay, decay_factor=0.9):
    """
    Creates an AdamW optimizer with Layer-Wise Learning Rate Decay (LLRD).
    """
    # DeBERTa-v3-large specific layer naming
    # Embeddings -> encoder.layer.0 ... encoder.layer.23 -> pooler/head

    named_parameters = list(model.named_parameters())
    optimizer_grouped_parameters = []

    # Define layers
    # DeBERTa usually has 'embeddings', 'encoder.layer.X', and the custom head

    # Base LR for the task-specific head
    head_lr = learning_rate

    # Parameters in the custom head (fc)
    head_params = [p for n, p in named_parameters if "backbone" not in n]
    optimizer_grouped_parameters.append(
        {"params": head_params, "weight_decay": weight_decay, "lr": head_lr}
    )

    # Backbone parameters
    # We group them by layer index
    # 24 layers in large model
    num_layers = 24

    for layer_i in range(num_layers - 1, -1, -1):
        layer_lr = head_lr * (decay_factor ** (num_layers - layer_i))

        # Filter params for this layer
        layer_params = [
            p for n, p in named_parameters if f"encoder.layer.{layer_i}." in n
        ]

        if layer_params:
            optimizer_grouped_parameters.append(
                {"params": layer_params, "weight_decay": weight_decay, "lr": layer_lr}
            )

    # Embeddings and other initial layers get the lowest LR
    embeddings_lr = head_lr * (decay_factor ** (num_layers + 1))
    embeddings_params = [
        p
        for n, p in named_parameters
        if "embeddings" in n
        or "rel_embeddings" in n
        or "LayerNorm" in n
        and "encoder" not in n
    ]

    if embeddings_params:
        optimizer_grouped_parameters.append(
            {
                "params": embeddings_params,
                "weight_decay": weight_decay,
                "lr": embeddings_lr,
            }
        )

    return torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6)


def train_transformer_expert(load_cached_data=True):
    """
    Main function to train the Transformer Expert (DeBERTa) with 5-fold CV.
    Generates OOF predictions and Test predictions.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed predictions.

    Returns:
        tuple: (oof_preds, test_preds)
            oof_preds: np.array of shape (N_train, 3)
            test_preds: np.array of shape (N_test, 3)
    """
    seed_everything(Config.SEED)

    # Paths for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    oof_cache_path = os.path.join(Config.WORKING_DIR, "oof_transformer.npy")
    test_cache_path = os.path.join(Config.WORKING_DIR, "test_preds_transformer.npy")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(oof_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print("Loading cached Transformer predictions...")
        try:
            oof_preds = np.load(oof_cache_path)
            test_preds = np.load(test_cache_path)
            return oof_preds, test_preds
        except Exception as e:
            print(f"Failed to load cache: {e}. Retraining...")

    print("Training Transformer Expert (DeBERTa-v3-Large)...")

    # Load Data
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Create Folds
    df_train = get_stratified_folds(
        df_train, num_folds=Config.NUM_FOLDS, seed=Config.SEED
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Placeholders
    oof_preds = np.zeros((len(df_train), Config.NUM_CLASSES))
    test_preds_fold_sum = np.zeros((len(df_test), Config.NUM_CLASSES))

    # Test Dataset (Common across folds)
    test_dataset = AuthorDataset(df_test, tokenizer, Config.MAX_LENGTH, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    device = Config.DEVICE

    for fold in range(Config.NUM_FOLDS):
        print(f"\n=== Fold {fold + 1}/{Config.NUM_FOLDS} ===")

        # Prepare Data
        train_idx = df_train[df_train["fold"] != fold].index.values
        val_idx = df_train[df_train["fold"] == fold].index.values

        train_sub = df_train.loc[train_idx].reset_index(drop=True)
        val_sub = df_train.loc[val_idx].reset_index(drop=True)

        train_dataset = AuthorDataset(
            train_sub, tokenizer, Config.MAX_LENGTH, is_test=False
        )
        val_dataset = AuthorDataset(
            val_sub, tokenizer, Config.MAX_LENGTH, is_test=False
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
        model = DebertaWithPooling(
            Config.MODEL_NAME, Config.NUM_CLASSES, Config.HIDDEN_DROPOUT_PROB
        )
        model.to(device)

        # Optimizer & Scheduler
        optimizer = get_llrd_optimizer(
            model,
            learning_rate=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            decay_factor=Config.LLRD_DECAY,
        )

        num_training_steps = int(
            len(train_loader) * Config.EPOCHS / Config.GRADIENT_ACCUMULATION_STEPS
        )
        num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        criterion = nn.CrossEntropyLoss()

        # Training Loop
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss = 0.0
            scaler = torch.cuda.amp.GradScaler()

            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), Config.MAX_GRAD_NORM
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            val_loss_accum = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids, attention_mask)
                        loss = criterion(outputs, labels)

                    val_loss_accum += loss.item()
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()

                    val_preds.append(probs)
                    val_labels.append(labels.cpu().numpy())

            val_preds = np.concatenate(val_preds)
            val_labels = np.concatenate(val_labels)

            # Calculate metrics
            avg_val_loss = val_loss_accum / len(val_loader)
            score = compute_log_loss(val_labels, val_preds)

            print(
                f"Epoch {epoch+1}: Train Loss={avg_train_loss:.5f}, Val Loss={avg_val_loss:.5f}, LogLoss={score:.10f}"
            )

            # Early Stopping Check
            if score < best_loss:
                best_loss = score
                patience_counter = 0
                best_model_state = model.state_dict()

                # Store OOF predictions for best epoch
                oof_preds[val_idx] = val_preds
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        # Inference on Test Set with Best Model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        fold_test_preds = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask)

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                fold_test_preds.append(probs)

        fold_test_preds = np.concatenate(fold_test_preds)
        test_preds_fold_sum += fold_test_preds

        # Cleanup
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

    # Average Test Predictions
    test_preds = test_preds_fold_sum / Config.NUM_FOLDS

    # Save to Cache
    try:
        np.save(oof_cache_path, oof_preds)
        np.save(test_cache_path, test_preds)
        print("Transformer predictions cached.")
    except Exception as e:
        print(f"Warning: Failed to save cache. {e}")

    return oof_preds, test_preds
