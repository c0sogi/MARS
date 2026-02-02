import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import compute_map5, predict_knn


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the Siamese Network for one epoch.

    Args:
        model (nn.Module): The SiameseNet model.
        dataloader (DataLoader): DataLoader for SiameseWhaleDataset.
        criterion (nn.Module): Loss function (ContrastiveLoss).
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run training on.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, ((img1, img2), labels) in enumerate(dataloader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # SiameseNet returns (output1, output2)
        out1, out2 = model(img1, img2)

        # Compute loss
        loss = criterion(out1, out2, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Aggregate loss
        # loss.item() is the average loss for the batch (default reduction='mean')
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return epoch_loss


def evaluate_loss(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set using Contrastive Loss.

    Args:
        model (nn.Module): The SiameseNet model.
        dataloader (DataLoader): DataLoader for validation (must yield pairs).
        criterion (nn.Module): Loss function.
        device (torch.device): Device.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, ((img1, img2), labels) in enumerate(dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return epoch_loss


def extract_embeddings(model, dataloader, device):
    """
    Passes images through the network to generate embeddings.
    Used for building the reference database and for inference.

    Args:
        model (nn.Module): The model (SiameseNet or EmbeddingNet).
        dataloader (DataLoader): DataLoader yielding (image, id, filename).
        device (torch.device): Device.

    Returns:
        dict: 'embeddings': np.array (N, D), 'ids': list of ids, 'images': list of filenames
    """
    model.eval()

    # If model is SiameseNet, use the underlying embedding_net
    if hasattr(model, "embedding_net"):
        net = model.embedding_net
    else:
        net = model

    all_embeddings = []
    all_ids = []
    all_filenames = []

    with torch.no_grad():
        for imgs, ids, filenames in dataloader:
            imgs = imgs.to(device)

            # Forward pass through EmbeddingNet
            embeddings = net(imgs)

            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend(ids)
            all_filenames.extend(filenames)

    if len(all_embeddings) > 0:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        all_embeddings = np.array([])

    return {"embeddings": all_embeddings, "ids": all_ids, "images": all_filenames}


def train_model(
    model,
    train_loader,
    val_loader,
    train_eval_loader,
    val_eval_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=Config.NUM_EPOCHS,
    patience=Config.PATIENCE,
    save_path=Config.MODEL_SAVE_PATH,
):
    """
    Full training loop with Early Stopping based on MAP@5.
    Cite solution_lesson_node_00002: Validation Strategy for Metric Learning.
    """
    best_map5 = 0.0
    epochs_no_improve = 0

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate Loss (for monitoring)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        # Validate MAP@5 (for Early Stopping)
        # Extract embeddings
        train_res = extract_embeddings(model, train_eval_loader, device)
        val_res = extract_embeddings(model, val_eval_loader, device)

        # Predict
        val_preds = predict_knn(
            val_res["embeddings"],
            train_res["embeddings"],
            np.array(train_res["ids"]),
            threshold=Config.NEW_WHALE_THRESHOLD,
        )

        # Compute Metric
        val_map5 = compute_map5(val_res["ids"], val_preds)

        # Step Scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # Use loss for scheduler as it's smoother
            else:
                scheduler.step()

        print(
            f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val MAP@5: {val_map5:.5f}"
        )

        # Early Stopping Check (Maximize MAP@5)
        if val_map5 > best_map5:
            best_map5 = val_map5
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print("Training complete.")

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Best model weights loaded.")

    return model
