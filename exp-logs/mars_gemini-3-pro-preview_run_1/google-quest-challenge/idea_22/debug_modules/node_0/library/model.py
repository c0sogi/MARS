import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader

from library.dataset import StackExchangeDataset, collate_fn
from library.utils import (
    seed_everything,
    compute_spearman_metric,
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
)


class PartitionedPoolingDualEncoder(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=30):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Feature dimension from backbone
        self.hidden_size = self.config.hidden_size

        # Input feature dimension calculation
        # Raw Pools: u_title, u_body, v (3 vectors)
        # Interactions: u_title*v, |u_title-v|, u_body*v, |u_body-v| (4 vectors)
        # Max Pools: max_title, max_body, max_answer (3 vectors)
        # Total vectors: 10
        self.feature_dim = self.hidden_size * 10

        self.layer_norm = nn.LayerNorm(self.feature_dim)

        # Residual Interaction Head
        # Structure: Output = Linear(Concat(F, Dropout(ReLU(Linear(F)))))
        self.projection_dim = 1024
        self.head_proj = nn.Linear(self.feature_dim, self.projection_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.head_out = nn.Linear(self.feature_dim + self.projection_dim, num_labels)

    def _mean_pooling(self, hidden_states, mask):
        # mask: (bs, seq_len) -> (bs, seq_len, 1)
        mask_expanded = mask.unsqueeze(-1)
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def _max_pooling(self, hidden_states, mask):
        mask_expanded = mask.unsqueeze(-1)
        # Set masked values to a very small number so they aren't selected as max
        hidden_states = hidden_states.clone()
        hidden_states[mask_expanded == 0] = -1e9
        max_embeddings = torch.max(hidden_states, dim=1)[0]
        return max_embeddings

    def forward(
        self,
        input_ids_q,
        attention_mask_q,
        title_mask,
        body_mask,
        input_ids_a,
        attention_mask_a,
    ):
        # Question Branch
        out_q = self.backbone(input_ids=input_ids_q, attention_mask=attention_mask_q)
        hidden_q = out_q.last_hidden_state

        # Answer Branch
        out_a = self.backbone(input_ids=input_ids_a, attention_mask=attention_mask_a)
        hidden_a = out_a.last_hidden_state

        # Partitioned Pooling (Question)
        u_title = self._mean_pooling(hidden_q, title_mask)
        u_body = self._mean_pooling(hidden_q, body_mask)
        max_title = self._max_pooling(hidden_q, title_mask)
        max_body = self._max_pooling(hidden_q, body_mask)

        # Pooling (Answer)
        v = self._mean_pooling(hidden_a, attention_mask_a)
        max_answer = self._max_pooling(hidden_a, attention_mask_a)

        # Interactions
        i_intent_prod = u_title * v
        i_intent_diff = torch.abs(u_title - v)
        i_context_prod = u_body * v
        i_context_diff = torch.abs(u_body - v)

        # Concatenation
        features = torch.cat(
            [
                u_title,
                u_body,
                v,
                i_intent_prod,
                i_intent_diff,
                i_context_prod,
                i_context_diff,
                max_title,
                max_body,
                max_answer,
            ],
            dim=1,
        )

        # Normalization
        features = self.layer_norm(features)

        # Residual Head
        proj = self.head_proj(features)
        proj = self.activation(proj)
        proj = self.dropout(proj)

        # Skip connection: Concat raw features with processed features
        concat_out = torch.cat([features, proj], dim=1)
        logits = self.head_out(concat_out)

        return logits


def validate(model, dataloader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_q = batch["input_ids_q"].to(device)
            attention_mask_q = batch["attention_mask_q"].to(device)
            title_mask = batch["title_mask"].to(device)
            body_mask = batch["body_mask"].to(device)
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids_q,
                attention_mask_q,
                title_mask,
                body_mask,
                input_ids_a,
                attention_mask_a,
            )

            # Sigmoid for prediction
            probs = torch.sigmoid(logits)

            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    return compute_spearman_metric(targets, preds)


def generate_submission(model, working_dir, device, model_name, debug):
    # Load best model
    best_path = os.path.join(working_dir, "best_model.pth")
    if not os.path.exists(best_path):
        print(f"Best model not found at {best_path}. Skipping submission generation.")
        return

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_dataset = StackExchangeDataset(
        split="test", tokenizer_name=model_name, cache_dir=working_dir, debug=debug
    )
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    all_preds = []
    all_qa_ids = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids_q = batch["input_ids_q"].to(device)
            attention_mask_q = batch["attention_mask_q"].to(device)
            title_mask = batch["title_mask"].to(device)
            body_mask = batch["body_mask"].to(device)
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)

            logits = model(
                input_ids_q,
                attention_mask_q,
                title_mask,
                body_mask,
                input_ids_a,
                attention_mask_a,
            )
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_qa_ids.extend(batch["qa_id"])

    all_preds = np.concatenate(all_preds, axis=0)

    # Create submission DF
    sample_sub = pd.read_csv("./input/sample_submission.csv")
    target_cols = [c for c in sample_sub.columns if c != "qa_id"]

    sub_df = pd.DataFrame(all_preds, columns=target_cols)
    sub_df.insert(0, "qa_id", all_qa_ids)

    # Ensure directory exists
    os.makedirs("./submission", exist_ok=True)
    sub_df.to_csv("./submission/submission.csv", index=False)
    print("Submission saved to ./submission/submission.csv")


def train_model(
    epochs=7,
    stop_epoch=3,
    batch_size=8,
    accum_steps=2,
    debug=False,
    working_dir="./working/idea_22",
    model_name="roberta-base",
):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(working_dir, exist_ok=True)

    # --- Data Loading ---
    print("Initializing Datasets...")
    train_dataset = StackExchangeDataset(
        split="train", tokenizer_name=model_name, cache_dir=working_dir, debug=debug
    )
    val_dataset = StackExchangeDataset(
        split="val", tokenizer_name=model_name, cache_dir=working_dir, debug=debug
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # --- Model Setup ---
    print("Initializing Model...")
    model = PartitionedPoolingDualEncoder(model_name=model_name)
    model.to(device)

    # Differential Learning Rates
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": 2e-5,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": 1e-3,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Phantom Scheduling
    # Total steps based on 'epochs' (7), but we stop at 'stop_epoch' (3)
    num_update_steps_per_epoch = len(train_loader) // accum_steps
    max_train_steps = num_update_steps_per_epoch * epochs

    # Warmup ratio 10%
    num_warmup_steps = int(0.1 * max_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
    )

    criterion = nn.BCEWithLogitsLoss()

    best_score = -1.0

    # --- Training Loop ---
    print("Starting Training...")
    for epoch in range(1, stop_epoch + 1):
        model.train()

        # Head Warmup: Freeze backbone in Epoch 1
        if epoch == 1:
            print("Epoch 1: Freezing Backbone for Head Warmup")
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            if epoch == 2:
                print("Epoch 2: Unfreezing Backbone")
            for param in model.backbone.parameters():
                param.requires_grad = True

        train_loss = AverageMeter()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # Move to device
            input_ids_q = batch["input_ids_q"].to(device)
            attention_mask_q = batch["attention_mask_q"].to(device)
            title_mask = batch["title_mask"].to(device)
            body_mask = batch["body_mask"].to(device)
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids_q,
                attention_mask_q,
                title_mask,
                body_mask,
                input_ids_a,
                attention_mask_a,
            )

            loss = criterion(logits, labels)

            # Gradient Accumulation
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss.update(loss.item() * accum_steps, labels.size(0))

        # Validation
        val_score = validate(model, val_loader, device)
        print(
            f"Epoch {epoch} | Train Loss: {train_loss.avg:.6f} | Val Spearman: {val_score:.6f}"
        )

        is_best = val_score > best_score
        if is_best:
            best_score = val_score

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_score": best_score,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            checkpoint_dir=working_dir,
        )

    print(f"Training finished. Best Val Score: {best_score:.6f}")

    # --- Inference ---
    print("Generating Submission...")
    generate_submission(model, working_dir, device, model_name, debug)
