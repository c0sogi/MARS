import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from library.utils import seed_everything, compute_spearmanr
from library.data import prepare_loaders
from library.model import LoRADebertaDualEncoder

# Constants
WORKING_DIR = "./working/idea_15/"
SUBMISSION_DIR = "./submission/"
CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")


def train_fn(
    model, dataloader, optimizer, scheduler, device, loss_fn, accumulation_steps=1
):
    model.train()
    total_loss = 0.0

    # Ensure gradients are zeroed at the start
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        # Move batch to device
        q_input_ids = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        a_input_ids = batch["a_input_ids"].to(device)
        a_attention_mask = batch["a_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        logits = model(
            q_input_ids=q_input_ids,
            q_attention_mask=q_attention_mask,
            a_input_ids=a_input_ids,
            a_attention_mask=a_attention_mask,
        )

        loss = loss_fn(logits, labels)

        # Normalize loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Step optimizer and scheduler only after accumulation steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # We multiply by accumulation_steps to log the "real" loss for this batch
        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)


def eval_fn(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                q_input_ids=q_input_ids,
                q_attention_mask=q_attention_mask,
                a_input_ids=a_input_ids,
                a_attention_mask=a_attention_mask,
            )

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # For Spearman, we can use logits directly or probabilities.
            # Using probabilities (sigmoid) is standard for the final output range [0,1].
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    avg_loss = total_loss / len(dataloader)
    score = compute_spearmanr(all_preds, all_labels)

    return avg_loss, score


def predict_fn(model, dataloader, device):
    model.eval()
    all_preds = []
    all_qa_ids = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            qa_ids = batch["qa_id"]

            logits = model(
                q_input_ids=q_input_ids,
                q_attention_mask=q_attention_mask,
                a_input_ids=a_input_ids,
                a_attention_mask=a_attention_mask,
            )

            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_qa_ids.extend(qa_ids.numpy())

    return np.concatenate(all_preds, axis=0), np.array(all_qa_ids)


def run_training(
    epochs=10, batch_size=16, lr=1e-3, weight_decay=0.01, patience=3, seed=42
):
    # Setup
    seed_everything(seed)
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # Data Loading
    train_loader, val_loader, test_loader, target_cols = prepare_loaders(
        load_cached_data=True, batch_size=batch_size, seed=seed
    )

    # Model Initialization
    model = LoRADebertaDualEncoder(num_labels=len(target_cols))
    model.to(device)

    # Optimizer with parameter groups
    # Exclude bias and LayerNorm from weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Scheduler
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training Loop
    best_score = -1.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, device, loss_fn
        )
        val_loss, val_score = eval_fn(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Spearman: {val_score}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"New best model saved with score: {best_score}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    test_preds, test_ids = predict_fn(model, test_loader, device)

    # Create Submission
    submission_df = pd.DataFrame(test_preds, columns=target_cols)
    submission_df.insert(0, "qa_id", test_ids)

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

    return best_score
