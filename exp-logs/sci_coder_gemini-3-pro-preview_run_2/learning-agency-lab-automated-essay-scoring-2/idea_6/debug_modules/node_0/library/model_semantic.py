import os
import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
from library.config import Config
from library.utils import AWP, quadratic_weighted_kappa


class DebertaV3Regressor(nn.Module):
    """
    Semantic Branch Model: DeBERTa-v3-large with Concatenated Mean and Max Pooling.
    """

    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.MODEL_BACKBONE)
        self.backbone = AutoModel.from_pretrained(
            Config.MODEL_BACKBONE, config=self.config
        )

        # The output dimension is hidden_size * 2 because we concatenate Mean and Max pooling
        self.fc = nn.Linear(self.config.hidden_size * 2, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize the weights of the regression head.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Forward pass with custom pooling.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # Shape: (Batch, Seq_Len, Hidden)

        # Expand attention_mask to match hidden state dimensions
        # mask shape: (Batch, Seq_Len) -> (Batch, Seq_Len, 1) -> (Batch, Seq_Len, Hidden)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # --- Concatenated Mean and Max Pooling ---

        # Mean Pooling
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # Max Pooling
        # Replace padding tokens with a very small number so they are not picked by max()
        # Clone to avoid modifying the original tensor in-place which might affect gradients if reused
        hidden_state_for_max = last_hidden_state.clone()
        hidden_state_for_max[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(hidden_state_for_max, 1)[0]

        # Concatenate
        concat_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)

        # Regression Head
        logits = self.fc(concat_embeddings)

        return logits.squeeze(-1)  # Return shape (Batch,)


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, device, epoch, awp=None, scaler=None
):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_size = input_ids.size(0)

        # Mixed Precision Training
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

        # Backward Pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Adversarial Weight Perturbation (AWP)
        if Config.USE_AWP and awp is not None and epoch >= Config.AWP_START_EPOCH:
            awp.attack_step(epoch)

            with torch.cuda.amp.autocast(enabled=True):
                outputs_adv = model(input_ids, attention_mask)
                loss_adv = criterion(outputs_adv, labels)

            if scaler:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()

            awp.restore(epoch)

        # Optimizer Step
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Validation loop for one epoch. Returns loss, predictions, and QWK score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            batch_size = input_ids.size(0)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            preds.append(outputs.float().cpu().numpy())
            targets.append(labels.float().cpu().numpy())

    epoch_loss = running_loss / dataset_size
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    qwk = quadratic_weighted_kappa(targets, preds)

    return epoch_loss, preds, qwk


def train_semantic_fold(fold_idx, train_loader, val_loader):
    """
    Orchestrates the training for a single fold.
    """
    print(f"\n=== Training Semantic Model | Fold {fold_idx} ===")

    device = Config.DEVICE
    model = DebertaV3Regressor()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),  # 10% warmup
        num_training_steps=num_train_steps,
    )

    # Loss Function
    if Config.LOSS_FN == "SmoothL1Loss":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    # AWP
    awp = None
    if Config.USE_AWP:
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.AWP_LR,
            adv_eps=Config.AWP_EPS,
            start_epoch=Config.AWP_START_EPOCH,
        )

    # Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Training Loop
    best_qwk = -1.0
    best_loss = float("inf")
    early_stopping_counter = 0
    patience = 3  # Early stopping patience

    save_path = os.path.join(Config.MODEL_DIR, f"deberta_fold_{fold_idx}.bin")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            awp,
            scaler,
        )

        val_loss, val_preds, val_qwk = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val QWK: {val_qwk}"
        )

        # Save best model based on QWK
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            print(f"Score Improved ({best_qwk}). Saving model to {save_path}...")
            torch.save(model.state_dict(), save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs with no improvement."
            )
            break

    # Cleanup
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return best_qwk


def predict_semantic(model, loader, device):
    """
    Generates predictions for a given loader using the provided model.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(input_ids, attention_mask)

            preds.append(outputs.float().cpu().numpy())

    return np.concatenate(preds)
