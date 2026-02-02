import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoConfig, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from library.config import Config
from library.utils import (
    seed_everything,
    get_device,
    save_checkpoint,
    load_checkpoint,
    print_metrics,
    format_time,
)
from library.data_processing import (
    load_data,
    create_dataloader,
    get_tokenizer,
)


class HybridDeberta(nn.Module):
    """
    Hybrid Architecture combining DeBERTa-v3 [CLS] embedding with
    Structural SVD features, using Variable-Rate Multi-Sample Dropout.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config

        # Load Backbone
        model_config = AutoConfig.from_pretrained(self.config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(
            self.config.MODEL_NAME, config=model_config
        )

        # Structural Feature Processing
        # LayerNorm for SVD features (dim=256)
        self.svd_norm = nn.LayerNorm(self.config.SVD_DIM)

        # Fusion Dimension: Backbone Hidden (768) + SVD Dim (256) = 1024
        self.fusion_dim = model_config.hidden_size + self.config.SVD_DIM

        # Variable-Rate Multi-Sample Dropout (VR-MSD)
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p) for p in self.config.DROPOUT_RATES]
        )

        # Classification Head (Shared)
        self.fc = nn.Linear(self.fusion_dim, 1)

        # Initialization
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.MODEL_NAME == "microsoft/deberta-v3-base"
                and 0.02
                or 0.02,
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, svd_feat):
        # Backbone Forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [Batch, 768]

        # Structural Feature Normalization
        svd_normed = self.svd_norm(svd_feat)  # [Batch, 256]

        # Feature Fusion
        fused_features = torch.cat([cls_embedding, svd_normed], dim=1)  # [Batch, 1024]

        # VR-MSD Forward
        logits_list = []
        for dropout in self.dropouts:
            dropped = dropout(fused_features)
            logits_list.append(self.fc(dropped))

        # Average Logits
        logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)

        return logits


class AWP:
    """
    Adversarial Weight Perturbation implementation.
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1e-4, adv_eps=1e-4):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
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
                    self.backup[name] = param.data.clone()
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    use_awp=False,
    awp=None,
):
    model.train()
    scaler = GradScaler()
    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        svd_feat = batch["svd_feat"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        batch_size = input_ids.size(0)

        with autocast():
            logits = model(input_ids, attention_mask, svd_feat)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        if use_awp and awp is not None:
            scaler.unscale_(optimizer)
            # AWP Attack
            awp.attack()
            with autocast():
                logits_adv = model(input_ids, attention_mask, svd_feat)
                loss_adv = criterion(logits_adv, labels)

            scaler.scale(loss_adv).backward()
            awp.restore()
        else:
            scaler.unscale_(optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        pbar.set_postfix(loss=running_loss / dataset_size)

    return running_loss / dataset_size


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    preds = []
    targets = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        svd_feat = batch["svd_feat"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        batch_size = input_ids.size(0)

        with autocast():
            logits = model(input_ids, attention_mask, svd_feat)
            loss = criterion(logits, labels)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
        targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    try:
        auc = roc_auc_score(targets, preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc, preds


@torch.no_grad()
def inference(model, dataloader, device):
    model.eval()
    preds = []

    for batch in tqdm(dataloader, desc="Inference", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        svd_feat = batch["svd_feat"].to(device)

        with autocast():
            logits = model(input_ids, attention_mask, svd_feat)

        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)

    return np.concatenate(preds)


def run_training_fold(
    fold, train_df, val_df, train_svd, val_svd, tokenizer, device, stage_name="Teacher"
):
    print(f"\n[{stage_name}] Starting Fold {fold}")

    # DataLoaders
    train_loader = create_dataloader(
        train_df, train_svd, tokenizer, is_train=True, shuffle=True
    )
    val_loader = create_dataloader(
        val_df, val_svd, tokenizer, is_train=False, shuffle=False
    )

    # Model
    model = HybridDeberta().to(device)

    # Optimizer (Differential Learning Rates)
    optimizer_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": Config.LR_BACKBONE,
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": Config.LR_HEAD,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_parameters)

    # Scheduler
    num_train_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Loss & AWP
    criterion = nn.BCEWithLogitsLoss()
    awp = (
        AWP(model, optimizer, adv_lr=Config.AWP_LR, adv_eps=Config.AWP_EPS)
        if Config.USE_AWP
        else None
    )

    best_auc = 0.0
    best_model_path = os.path.join(
        Config.WORKING_DIR, f"model_{stage_name.lower()}_fold_{fold}.bin"
    )

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        use_awp_epoch = Config.USE_AWP and (epoch >= Config.AWP_START_EPOCH)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            use_awp=use_awp_epoch,
            awp=awp,
        )

        # Validate
        val_loss, val_auc, _ = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch} | Time: {format_time(elapsed)} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model.state_dict(), best_model_path)
            print(f"  >>> New Best AUC! Model saved to {best_model_path}")

    # Load best model to return
    model.load_state_dict(load_checkpoint(best_model_path, device))
    return model, best_auc


def run_pipeline():
    seed_everything(Config.SEED)
    device = get_device()
    tokenizer = get_tokenizer()

    # 1. Load Data
    raw_train, raw_val, test_df, raw_train_svd, raw_val_svd, test_svd = load_data(
        load_cached_data=True, debug=Config().debug
    )

    # Merge Train and Val for Cross-Validation
    full_train_df = pd.concat([raw_train, raw_val]).reset_index(drop=True)
    full_train_svd = np.vstack([raw_train_svd, raw_val_svd])

    # Prepare K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # ==========================================
    # STAGE 1: Teacher Training
    # ==========================================
    print("\n" + "=" * 40)
    print(" STAGE 1: Teacher Training ")
    print("=" * 40)

    teacher_preds = np.zeros((len(test_df), 1))

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_train_df, full_train_df["Insult"])
    ):
        # Split Data
        train_fold_df = full_train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = full_train_df.iloc[val_idx].reset_index(drop=True)

        train_fold_svd = full_train_svd[train_idx]
        val_fold_svd = full_train_svd[val_idx]

        # Train Teacher
        model, _ = run_training_fold(
            fold,
            train_fold_df,
            val_fold_df,
            train_fold_svd,
            val_fold_svd,
            tokenizer,
            device,
            stage_name="Teacher",
        )

        # Inference on Test (Accumulate for averaging)
        test_loader = create_dataloader(
            test_df, test_svd, tokenizer, is_train=False, is_test=True, shuffle=False
        )
        fold_preds = inference(model, test_loader, device)
        teacher_preds += fold_preds

        # Clean up
        del model, train_fold_df, val_fold_df, train_fold_svd, val_fold_svd
        torch.cuda.empty_cache()
        gc.collect()

    # Average Teacher Predictions
    teacher_preds /= Config.N_FOLDS

    # Save Teacher Predictions
    np.save(Config.TEACHER_PREDS_PATH, teacher_preds)

    # ==========================================
    # STAGE 2: Pseudo-Labeling & Student Training
    # ==========================================
    print("\n" + "=" * 40)
    print(" STAGE 2: Pseudo-Labeling & Student Training ")
    print("=" * 40)

    # Generate Pseudo Labels
    # High confidence: > 0.9 (Insult), < 0.1 (Neutral)
    pseudo_mask = (teacher_preds >= Config.PSEUDO_LABEL_HIGH) | (
        teacher_preds <= Config.PSEUDO_LABEL_LOW
    )
    pseudo_mask = pseudo_mask.flatten()

    pseudo_df = test_df.iloc[pseudo_mask].copy().reset_index(drop=True)
    pseudo_svd = test_svd[pseudo_mask]

    # Assign labels (hard labels based on threshold)
    pseudo_labels = (teacher_preds[pseudo_mask] >= 0.5).astype(int).flatten()
    pseudo_df["Insult"] = pseudo_labels

    print(f"Generated {len(pseudo_df)} pseudo-labels from {len(test_df)} test samples.")

    # Augment Training Data
    augmented_train_df = pd.concat([full_train_df, pseudo_df]).reset_index(drop=True)
    augmented_train_svd = np.vstack([full_train_svd, pseudo_svd])

    # Student Training Loop
    student_preds = np.zeros((len(test_df), 1))

    # We use the same SKF split on the ORIGINAL training data to ensure validation consistency
    # But we append the pseudo-data to the training set of each fold

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_train_df, full_train_df["Insult"])
    ):
        # Base Split
        base_train_df = full_train_df.iloc[train_idx]
        val_fold_df = full_train_df.iloc[val_idx].reset_index(drop=True)

        base_train_svd = full_train_svd[train_idx]
        val_fold_svd = full_train_svd[val_idx]

        # Augment Training Fold
        train_fold_df = pd.concat([base_train_df, pseudo_df]).reset_index(drop=True)
        train_fold_svd = np.vstack([base_train_svd, pseudo_svd])

        # Train Student
        model, _ = run_training_fold(
            fold,
            train_fold_df,
            val_fold_df,
            train_fold_svd,
            val_fold_svd,
            tokenizer,
            device,
            stage_name="Student",
        )

        # Inference on Test
        test_loader = create_dataloader(
            test_df, test_svd, tokenizer, is_train=False, is_test=True, shuffle=False
        )
        fold_preds = inference(model, test_loader, device)
        student_preds += fold_preds

        # Clean up
        del model, train_fold_df, val_fold_df, train_fold_svd, val_fold_svd
        torch.cuda.empty_cache()
        gc.collect()

    # Average Student Predictions
    student_preds /= Config.N_FOLDS

    # ==========================================
    # Submission
    # ==========================================
    submission_df = pd.DataFrame(
        {
            "id": range(
                len(test_df)
            ),  # Assuming implicit ID based on row index if not present
            "prediction": student_preds.flatten(),
        }
    )

    # The sample submission format in the description implies just the prediction column might be needed or specific format
    # "Your predictions should be a number in the range [0,1]. See 'sample_submissions_null.csv' for the correct format."
    # Usually we overwrite the sample submission.

    try:
        sample_sub = pd.read_csv("./input/sample_submission_null.csv")
        # Check if sample submission has specific columns
        if "Insult" in sample_sub.columns:
            sample_sub["Insult"] = student_preds.flatten()
            sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        else:
            # Fallback
            submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    except Exception:
        # Fallback if sample file read fails
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")


import gc

if __name__ == "__main__":
    run_pipeline()
