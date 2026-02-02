import torch
import numpy as np
from library.config import Config
from library.utils import AverageMeter, jaccard, get_selected_text
from library.loss import TweetLoss


def train_fn(dataloader, model, optimizer, device, scheduler):
    """
    Executes one training epoch.

    Args:
        dataloader (DataLoader): The training data loader.
        model (nn.Module): The neural network model.
        optimizer (Optimizer): The optimizer.
        device (str): Device to run training on ('cuda' or 'cpu').
        scheduler (Scheduler): Learning rate scheduler.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()
    criterion = TweetLoss()

    for data in dataloader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        start_positions = data["start_positions"].to(device)
        end_positions = data["end_positions"].to(device)

        optimizer.zero_grad()

        # Forward pass
        start_logits, end_logits = model(input_ids, attention_mask)

        # Calculate loss
        loss = criterion(start_logits, end_logits, start_positions, end_positions)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer and Scheduler steps
        optimizer.step()
        scheduler.step()

        # Update metrics
        loss_meter.update(loss.item(), input_ids.size(0))

    return loss_meter.avg


def eval_fn(dataloader, model, device):
    """
    Executes evaluation on the validation set.

    Args:
        dataloader (DataLoader): The validation data loader.
        model (nn.Module): The neural network model.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (Average Loss, Average Jaccard Score)
    """
    model.eval()
    loss_meter = AverageMeter()
    jaccard_meter = AverageMeter()
    criterion = TweetLoss()

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            start_positions = data["start_positions"].to(device)
            end_positions = data["end_positions"].to(device)

            # Metadata for post-processing and metric calculation
            texts = data["text"]
            sentiments = data["sentiment"]
            # selected_text is present in validation data
            selected_texts = data.get("selected_text", [])
            offsets = data["offsets"].numpy()

            # Forward pass
            start_logits, end_logits = model(input_ids, attention_mask)

            # Loss calculation
            loss = criterion(start_logits, end_logits, start_positions, end_positions)
            loss_meter.update(loss.item(), input_ids.size(0))

            # Decode predictions
            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            # Iterate through batch to calculate Jaccard
            for i in range(len(texts)):
                sentiment = sentiments[i]
                orig_text = texts[i]
                target_text = selected_texts[i]
                offset = offsets[i]

                if sentiment == "neutral":
                    # Deterministic rule for neutral tweets
                    pred_text = orig_text
                else:
                    # Span selection logic for positive/negative
                    s_prob = start_probs[i]
                    e_prob = end_probs[i]

                    # Create a score matrix where score[j, k] = P_start(j) + P_end(k)
                    # Shape: (Seq_Len, Seq_Len)
                    score_mat = np.expand_dims(s_prob, 1) + np.expand_dims(e_prob, 0)

                    # Mask lower triangle to enforce start_index <= end_index
                    # np.triu keeps the upper triangle (including diagonal)
                    score_mat = np.triu(score_mat)

                    # Find the indices (start, end) that maximize the score
                    best_idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)
                    pred_start_idx = best_idx[0]
                    pred_end_idx = best_idx[1]

                    # Extract text using offsets
                    pred_text = get_selected_text(
                        orig_text, pred_start_idx, pred_end_idx, offset
                    )

                score = jaccard(pred_text, target_text)
                jaccard_meter.update(score, 1)

    return loss_meter.avg, jaccard_meter.avg


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    save_path,
    patience=2,
):
    """
    Orchestrates the training loop with Early Stopping.

    Args:
        model: The model to train.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        device: Device to use.
        epochs: Number of epochs to train.
        save_path: Path to save the best model.
        patience: Number of epochs to wait for improvement before stopping.
    """
    best_jaccard = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        val_loss, val_jaccard = eval_fn(val_loader, model, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Jaccard: {val_jaccard}")

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with Jaccard: {best_jaccard}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
