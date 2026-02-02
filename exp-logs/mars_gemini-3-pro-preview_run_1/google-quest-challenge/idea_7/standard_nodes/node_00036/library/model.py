import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoModel, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from library.config import Config, seed_everything
from library.utils import compute_spearman_metric
from library.data import get_dataloaders


# ==========================================
# Pooling Layer
# ==========================================
class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # mask: [batch, seq_len] -> [batch, seq_len, 1]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Mean Pooling
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # Max Pooling
        # Set padding tokens to large negative value so they aren't selected as max
        # Use masked_fill to avoid in-place modification
        last_hidden_state_masked = last_hidden_state.masked_fill(
            input_mask_expanded == 0, -1e9
        )
        max_embeddings = torch.max(last_hidden_state_masked, 1)[0]

        return mean_embeddings, max_embeddings


# ==========================================
# Model Architecture
# ==========================================
class MultiTaskDualEncoder(nn.Module):
    def __init__(self):
        super(MultiTaskDualEncoder, self).__init__()
        self.config = Config

        # Backbone
        self.backbone = AutoModel.from_pretrained(self.config.MODEL_NAME)

        # Pooling
        self.pooler = MeanMaxPooling()

        # Dimensions
        self.hidden_size = self.config.HIDDEN_SIZE
        # Pooled dim per branch = mean + max
        self.pooled_dim = self.hidden_size * 2

        # Fusion Dimension
        # Q_pool (2*H) + A_pool (2*H) + Q_mean*A_mean (H) + |Q_mean-A_mean| (H)
        self.fusion_dim = (self.pooled_dim * 2) + (self.hidden_size * 2)

        # Fusion Normalization
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

        # Main Head (Predicts 30 targets)
        self.main_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, len(self.config.TARGET_COLS)),
        )

        # Auxiliary Head (Predicts 21 Question targets from Q_pool only)
        self.aux_head = nn.Sequential(
            nn.Linear(self.pooled_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, len(self.config.QUESTION_TARGET_COLS)),
        )

        # Weight Initialization for Heads
        self._init_weights(self.main_head)
        self._init_weights(self.aux_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        # Encode Question
        q_out = self.backbone(input_ids=q_input_ids, attention_mask=q_attention_mask)
        q_mean, q_max = self.pooler(q_out.last_hidden_state, q_attention_mask)
        q_pool = torch.cat([q_mean, q_max], dim=1)  # [batch, 2*H]

        # Encode Answer
        a_out = self.backbone(input_ids=a_input_ids, attention_mask=a_attention_mask)
        a_mean, a_max = self.pooler(a_out.last_hidden_state, a_attention_mask)
        a_pool = torch.cat([a_mean, a_max], dim=1)  # [batch, 2*H]

        # Auxiliary Head Output (Question Only)
        aux_logits = self.aux_head(q_pool)

        # Interaction Features (on Mean vectors only)
        interaction_prod = q_mean * a_mean
        interaction_diff = torch.abs(q_mean - a_mean)

        # Fusion
        fused_vector = torch.cat(
            [q_pool, a_pool, interaction_prod, interaction_diff], dim=1
        )
        fused_vector = self.fusion_norm(fused_vector)

        # Main Head Output
        main_logits = self.main_head(fused_vector)

        return main_logits, aux_logits


# ==========================================
# Training Helper Functions
# ==========================================
def get_optimizer_params(model, lr_backbone, lr_head, weight_decay):
    # Separate backbone and head parameters
    backbone_params = list(model.backbone.named_parameters())
    head_params = (
        list(model.pooler.named_parameters())
        + list(model.fusion_norm.named_parameters())
        + list(model.main_head.named_parameters())
        + list(model.aux_head.named_parameters())
    )

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        # Backbone with decay
        {
            "params": [
                p for n, p in backbone_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": lr_backbone,
        },
        # Backbone without decay
        {
            "params": [
                p for n, p in backbone_params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr_backbone,
        },
        # Head with decay
        {
            "params": [
                p for n, p in head_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": lr_head,
        },
        # Head without decay
        {
            "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr_head,
        },
    ]
    return optimizer_grouped_parameters


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler):
    model.train()
    total_loss = 0

    loss_fn = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(dataloader):
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)
        a_ids = batch["a_input_ids"].to(device)
        a_mask = batch["a_attention_mask"].to(device)
        labels = batch["labels"].to(device)
        aux_labels = batch["aux_labels"].to(device)

        optimizer.zero_grad()

        with autocast():
            main_logits, aux_logits = model(q_ids, q_mask, a_ids, a_mask)

            loss_main = loss_fn(main_logits, labels)
            loss_aux = loss_fn(aux_logits, aux_labels)

            loss = loss_main + (Config.AUX_LOSS_WEIGHT * loss_aux)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.5f}")
    return avg_loss


def validate(model, dataloader, device):
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            a_ids = batch["a_input_ids"].to(device)
            a_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # We only use main logits for validation/prediction
            main_logits, _ = model(q_ids, q_mask, a_ids, a_mask)
            preds = torch.sigmoid(main_logits)

            preds_list.append(preds.cpu())
            targets_list.append(labels.cpu())

    preds_all = torch.cat(preds_list, dim=0).numpy()
    targets_all = torch.cat(targets_list, dim=0).numpy()

    score = compute_spearman_metric(targets_all, preds_all)
    return score


def predict(model, dataloader, device):
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in dataloader:
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            a_ids = batch["a_input_ids"].to(device)
            a_mask = batch["a_attention_mask"].to(device)

            main_logits, _ = model(q_ids, q_mask, a_ids, a_mask)
            preds = torch.sigmoid(main_logits)

            preds_list.append(preds.cpu())

    preds_all = torch.cat(preds_list, dim=0).numpy()
    return preds_all


# ==========================================
# Main Execution
# ==========================================
def run_training(debug=False):
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=debug
    )

    # Initialize Model
    model = MultiTaskDualEncoder()
    model.to(device)

    # Optimizer & Scheduler
    optimizer_params = get_optimizer_params(
        model, Config.LR_BACKBONE, Config.LR_HEAD, Config.WEIGHT_DECAY
    )
    optimizer = torch.optim.AdamW(optimizer_params)

    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    scaler = GradScaler()

    best_score = -1.0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, scaler
        )

        # Validate
        val_score = validate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Val Spearman: {val_score:.10f}")

        # Save Best
        if val_score > best_score:
            print(
                f"Score Improved ({best_score:.5f} -> {val_score:.5f}). Saving model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print(f"Training Complete. Best Val Score: {best_score:.10f}")

    # ==========================================
    # Inference on Test Set
    # ==========================================
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)

    print("Predicting on test set...")
    test_preds = predict(model, test_loader, device)

    # Create Submission
    # We need qa_id from test.csv
    test_df = pd.read_csv(Config.TEST_PATH)

    submission = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
    submission.insert(0, "qa_id", test_df["qa_id"])

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    # This block is just for local testing if run directly,
    # but the primary entry point is run_training() called by an external script.
    run_training(debug=False)
