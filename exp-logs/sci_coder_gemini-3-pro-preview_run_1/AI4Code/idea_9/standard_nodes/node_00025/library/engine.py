import torch
import torch.nn as nn
import numpy as np
from scipy.stats import kendalltau
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for batch in dataloader:
        # Move inputs to device
        code_emb = batch["code_embeddings"].to(device)
        code_lens = batch["code_lens"].to(device)
        code_mask = batch["code_padding_mask"].to(device)
        md_emb = batch["markdown_embeddings"].to(device)
        md_lens = batch["md_lens"].to(device)
        md_mask = batch["md_padding_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # logits: (Batch, Max_MD, Max_Code + 1)
        logits = model(code_emb, code_lens, code_mask, md_emb, md_lens, md_mask)

        # Flatten for CrossEntropyLoss
        # Logits: (B * M, C + 1) -> Flattened
        # Labels: (B * M) -> Flattened
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        loss = loss_fn(logits_flat, labels_flat)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and average Kendall Tau score.
    """
    model.eval()
    total_loss = 0.0
    total_tau = 0.0
    num_batches = 0
    num_samples = 0

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in dataloader:
            code_emb = batch["code_embeddings"].to(device)
            code_lens = batch["code_lens"].to(device)
            code_mask = batch["code_padding_mask"].to(device)
            md_emb = batch["markdown_embeddings"].to(device)
            md_lens = batch["md_lens"].to(device)
            md_mask = batch["md_padding_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(code_emb, code_lens, code_mask, md_emb, md_lens, md_mask)

            # Compute Loss
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = loss_fn(logits_flat, labels_flat)
            total_loss += loss.item()

            # Compute Metric (Kendall Tau)
            # 1. Softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (B, M, C+1)

            # 2. Compute Expected Index (Soft Rank)
            # Create range [0, 1, ..., C]
            max_c = probs.size(-1)
            indices = torch.arange(max_c, device=device).float()

            # Sum(p_i * i) -> (B, M)
            pred_scores = torch.sum(probs * indices, dim=-1)

            # 3. Calculate Kendall Tau per notebook
            # The dataset provides MD cells sorted by their ground truth rank.
            # So we compare predicted scores against [0, 1, 2, ...]
            pred_scores_np = pred_scores.cpu().numpy()
            md_lens_np = md_lens.cpu().numpy()

            batch_tau_sum = 0.0

            for i in range(len(pred_scores_np)):
                length = md_lens_np[i]

                # If less than 2 items, order is trivial (score 1.0)
                if length < 2:
                    batch_tau_sum += 1.0
                else:
                    # Get scores for valid MD cells
                    scores = pred_scores_np[i, :length]
                    # Ground truth is simply increasing integers because input is sorted by rank
                    ground_truth = np.arange(length)

                    tau, _ = kendalltau(ground_truth, scores)

                    # Handle NaN (happens if all scores are identical)
                    if np.isnan(tau):
                        tau = 0.0

                    batch_tau_sum += tau

            total_tau += batch_tau_sum
            num_batches += 1
            num_samples += len(pred_scores_np)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_tau = total_tau / num_samples if num_samples > 0 else 0.0

    return avg_loss, avg_tau


def train_model(
    model, train_loader, val_loader, optimizer, device, epochs, patience, save_path
):
    """
    Main training loop with early stopping.
    """
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_tau = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Kendall Tau: {val_tau:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
