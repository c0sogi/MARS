import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold

from library.config import Config
from library.utils import (
    set_seed,
    compute_spearman,
    save_numpy_array,
    load_numpy_array,
    save_checkpoint,
    load_checkpoint,
    get_artifact_path,
)
from library.dataset import prepare_supervised_data, StackExchangeDataset
from library.modeling import CustomBackbone


class FineTuningModel(nn.Module):
    """
    Wraps the CustomBackbone with a linear head for Supervised Fine-Tuning.
    """

    def __init__(self, model_name_or_path, num_labels):
        super().__init__()
        self.backbone = CustomBackbone(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.hidden_size = config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, q_mask, a_mask, labels=None):
        # Extract features: h_cls, h_q, h_a, h_diff
        features = self.backbone(input_ids, attention_mask, q_mask, a_mask)

        # Use CLS embedding for the auxiliary classification task
        logits = self.classifier(features["h_cls"])

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return logits, features, loss


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        q_mask = batch["q_mask"].to(device)
        a_mask = batch["a_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits, _, loss = model(input_ids, attention_mask, q_mask, a_mask, labels)

        loss.backward()

        # Gradient Accumulation could be implemented here if needed,
        # but Config.GRAD_ACCUM_STEPS is 1 by default.
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    targets = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            q_mask = batch["q_mask"].to(device)
            a_mask = batch["a_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _, loss = model(input_ids, attention_mask, q_mask, a_mask, labels)

            total_loss += loss.item()
            preds.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    avg_loss = total_loss / len(dataloader)

    score = compute_spearman(targets, preds)
    return avg_loss, score


def extract_features(model, dataloader, device):
    """
    Extracts concatenated features [h_cls, h_q, h_a, h_diff] for the dataset.
    """
    model.eval()
    all_features = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            q_mask = batch["q_mask"].to(device)
            a_mask = batch["a_mask"].to(device)

            _, feats, _ = model(input_ids, attention_mask, q_mask, a_mask, labels=None)

            # Concatenate topological features: [Batch, 4 * Hidden]
            # h_cls, h_q, h_a, h_diff
            concat_feats = torch.cat(
                [feats["h_cls"], feats["h_q"], feats["h_a"], feats["h_diff"]], dim=1
            )

            all_features.append(concat_feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def run_fold(
    fold_idx, train_ds, val_ds, holdout_ds, test_ds, model_init_path, model_alias
):
    """
    Trains a model for one fold and extracts features.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check for cached fold artifacts
    fold_oof_path = f"{model_alias}_fold{fold_idx}_oof.npy"
    fold_holdout_path = f"{model_alias}_fold{fold_idx}_holdout.npy"
    fold_test_path = f"{model_alias}_fold{fold_idx}_test.npy"

    cached_oof = load_numpy_array(fold_oof_path)
    cached_holdout = load_numpy_array(fold_holdout_path)
    cached_test = load_numpy_array(fold_test_path)

    if (
        cached_oof is not None
        and cached_holdout is not None
        and cached_test is not None
    ):
        print(f"Fold {fold_idx} artifacts found in cache. Skipping training.")
        return cached_oof, cached_holdout, cached_test

    print(f"\n--- Starting Fold {fold_idx} Training ---")

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Initialize Model
    model = FineTuningModel(model_init_path, num_labels=30)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_score = -1.0
    best_state = None

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        val_loss, val_score = evaluate(model, val_loader, device)

        print(
            f"Fold {fold_idx} | Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val Spearman: {val_score:.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            best_state = model.state_dict()

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Extract Features
    print(f"Extracting features for Fold {fold_idx}...")

    # 1. OOF Features (Val set of this fold)
    oof_features = extract_features(model, val_loader, device)

    # 2. Holdout Features (Global Validation Set)
    holdout_loader = DataLoader(
        holdout_ds, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=4
    )
    holdout_features = extract_features(model, holdout_loader, device)

    # 3. Test Features
    test_loader = DataLoader(
        test_ds, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_features = extract_features(model, test_loader, device)

    # Save to cache
    save_numpy_array(oof_features, fold_oof_path)
    save_numpy_array(holdout_features, fold_holdout_path)
    save_numpy_array(test_features, fold_test_path)

    # Cleanup
    del (
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        holdout_loader,
        test_loader,
    )
    torch.cuda.empty_cache()
    gc.collect()

    return oof_features, holdout_features, test_features


def run_fine_tuning(
    model_alias, base_model_name, dapt_path=None, load_cached_data=True
):
    """
    Main driver for Supervised Fine-Tuning.

    Args:
        model_alias (str): Identifier for the model (e.g., 'deberta', 'mpnet').
        base_model_name (str): HuggingFace model name.
        dapt_path (str, optional): Path to DAPT weights. If exists, overrides base_model_name.
        load_cached_data (bool): Whether to use cached data/features.
    """
    set_seed(Config.SEED)

    # Check if final output exists
    final_train_path = f"{model_alias}_train_features.npy"
    final_val_path = f"{model_alias}_val_features.npy"
    final_test_path = f"{model_alias}_test_features.npy"

    if load_cached_data:
        if (
            load_numpy_array(final_train_path) is not None
            and load_numpy_array(final_val_path) is not None
            and load_numpy_array(final_test_path) is not None
        ):
            print(
                f"Final features for {model_alias} already exist. Skipping fine-tuning."
            )
            return

    # Determine model initialization path
    model_init_path = base_model_name
    if dapt_path and os.path.exists(dapt_path):
        # Check if it looks like a model directory
        if os.path.exists(os.path.join(dapt_path, "config.json")):
            print(f"Using DAPT weights from: {dapt_path}")
            model_init_path = dapt_path
        else:
            print(
                f"DAPT path provided but not valid model dir. Using base: {base_model_name}"
            )

    # Load Data
    print("Loading datasets for fine-tuning...")
    full_train_ds = prepare_supervised_data("train", load_cached_data=load_cached_data)
    holdout_val_ds = prepare_supervised_data("val", load_cached_data=load_cached_data)
    test_ds = prepare_supervised_data("test", load_cached_data=load_cached_data)

    # Load metadata for grouping
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    groups = train_df["question_body"].fillna("UNKNOWN").values

    # K-Fold Split
    gkf = GroupKFold(n_splits=Config.N_FOLDS)

    # Placeholders for aggregation
    # We need to reconstruct the train set order.
    # GroupKFold indices are not sequential, so we fill a pre-allocated array.
    # The feature dimension is 4 * hidden_size.
    # We need to peek at hidden size.
    temp_config = AutoConfig.from_pretrained(model_init_path)
    hidden_dim = temp_config.hidden_size
    feature_dim = 4 * hidden_dim

    oof_train_features = np.zeros((len(full_train_ds), feature_dim), dtype=np.float32)
    avg_holdout_features = np.zeros(
        (len(holdout_val_ds), feature_dim), dtype=np.float32
    )
    avg_test_features = np.zeros((len(test_ds), feature_dim), dtype=np.float32)

    fold_count = 0

    # Iterate Folds
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
        # Create Subsets
        train_subset = Subset(full_train_ds, train_idx)
        val_subset = Subset(full_train_ds, val_idx)

        # Run Fold
        oof_feats, holdout_feats, test_feats = run_fold(
            fold_idx,
            train_subset,
            val_subset,
            holdout_val_ds,
            test_ds,
            model_init_path,
            model_alias,
        )

        # Aggregate
        oof_train_features[val_idx] = oof_feats
        avg_holdout_features += holdout_feats
        avg_test_features += test_feats
        fold_count += 1

    # Average the predictions
    avg_holdout_features /= fold_count
    avg_test_features /= fold_count

    # Save Final Features
    print(f"Saving aggregated features for {model_alias}...")
    save_numpy_array(oof_train_features, final_train_path)
    save_numpy_array(avg_holdout_features, final_val_path)
    save_numpy_array(avg_test_features, final_test_path)

    # Also save targets for convenience (aligned with train features)
    # The prepare_supervised_data saves targets separately, but we ensure they are available.
    if load_numpy_array(f"{model_alias}_train_targets.npy") is None:
        targets = full_train_ds.targets
        save_numpy_array(targets, f"{model_alias}_train_targets.npy")

    print(f"Fine-tuning and feature extraction for {model_alias} completed.")
