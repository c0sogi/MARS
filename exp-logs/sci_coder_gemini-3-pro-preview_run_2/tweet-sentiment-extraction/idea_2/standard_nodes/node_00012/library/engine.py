import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.utils import AverageMeter, jaccard


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Computes the sum of CrossEntropyLoss for start and end indices.
    Uses Label Smoothing to mitigate overfitting on noisy labels (Cite solution_lesson_node_00010).
    """
    loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    return start_loss + end_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Training loop for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for d in data_loader:
        input_ids = d["ids"].to(device)
        attention_mask = d["mask"].to(device)
        targets_start = d["targets_start"].to(device)
        targets_end = d["targets_end"].to(device)

        optimizer.zero_grad()

        start_logits, end_logits = model(input_ids, attention_mask)

        loss = loss_fn(start_logits, end_logits, targets_start, targets_end)
        loss.backward()

        optimizer.step()
        if scheduler:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device, df_data):
    """
    Evaluation loop for validation set.
    Calculates Loss and Jaccard Score.

    Args:
        data_loader: DataLoader for validation set.
        model: The model to evaluate.
        device: Torch device.
        df_data: DataFrame containing 'textID' and 'selected_text' for ground truth.
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    # Pre-process ground truth for fast lookup
    gt_map = {}
    if "selected_text" in df_data.columns:
        gt_map = dict(
            zip(df_data["textID"].astype(str), df_data["selected_text"].astype(str))
        )

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            targets_start = d["targets_start"].to(device)
            targets_end = d["targets_end"].to(device)

            # Forward pass
            start_logits, end_logits = model(input_ids, attention_mask)

            # Calculate Loss
            loss = loss_fn(start_logits, end_logits, targets_start, targets_end)
            losses.update(loss.item(), input_ids.size(0))

            # Decode predictions
            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            offsets = d["offsets"].numpy()
            ids = d["textID"]
            texts = d["text"]
            sentiments = d["sentiment"]

            for i in range(len(ids)):
                text_id = ids[i]
                text = texts[i]
                sentiment = sentiments[i]
                offset = offsets[i]
                selected_text = gt_map.get(text_id, "")

                # Heuristic: If neutral, predict full text
                if sentiment == "neutral":
                    pred_string = text
                else:
                    # Find best span (s, e) maximizing P(s) + P(e) with s <= e
                    start_p = start_probs[i]
                    end_p = end_probs[i]

                    # Create sum matrix
                    score_mat = start_p[:, None] + end_p[None, :]

                    # Mask lower triangle (ensure start <= end)
                    upper_tri_mask = np.triu(np.ones_like(score_mat))
                    masked_score = score_mat * upper_tri_mask
                    # Set invalid positions to a very small number
                    masked_score[upper_tri_mask == 0] = -1e9

                    # Get indices of max
                    max_idx = np.unravel_index(
                        np.argmax(masked_score), masked_score.shape
                    )
                    idx_start, idx_end = max_idx

                    # Decode to string using offsets
                    try:
                        char_start = offset[idx_start][0]
                        char_end = offset[idx_end][1]

                        # Handle case where predicted span points to special tokens (0,0)
                        if char_start == 0 and char_end == 0:
                            pred_string = text
                        else:
                            pred_string = text[char_start:char_end]
                    except Exception:
                        pred_string = text

                # Compute Jaccard
                score = jaccard(selected_text, pred_string)
                jaccards.update(score, 1)

    return losses.avg, jaccards.avg


def generate_submission(data_loader, model, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            offsets = d["offsets"].numpy()
            ids = d["textID"]
            texts = d["text"]
            sentiments = d["sentiment"]

            for i in range(len(ids)):
                text = texts[i]
                sentiment = sentiments[i]
                offset = offsets[i]

                if sentiment == "neutral":
                    pred_string = text
                else:
                    start_p = start_probs[i]
                    end_p = end_probs[i]
                    score_mat = start_p[:, None] + end_p[None, :]
                    upper_tri_mask = np.triu(np.ones_like(score_mat))
                    masked_score = score_mat * upper_tri_mask
                    masked_score[upper_tri_mask == 0] = -1e9

                    max_idx = np.unravel_index(
                        np.argmax(masked_score), masked_score.shape
                    )
                    idx_start, idx_end = max_idx

                    try:
                        char_start = offset[idx_start][0]
                        char_end = offset[idx_end][1]
                        if char_start == 0 and char_end == 0:
                            pred_string = text
                        else:
                            pred_string = text[char_start:char_end]
                    except Exception:
                        pred_string = text

                predictions.append({"textID": ids[i], "selected_text": pred_string})

    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    save_path,
    val_df,
    patience=3,
):
    """
    Orchestrates the training process with Early Stopping.
    """
    best_jaccard = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        val_loss, val_jaccard = eval_fn(val_loader, model, device, val_df)

        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Jaccard: {val_jaccard}")

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), save_path)
            print("Model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
