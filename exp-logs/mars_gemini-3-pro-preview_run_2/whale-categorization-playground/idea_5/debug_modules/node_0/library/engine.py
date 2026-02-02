import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import map_at_5
from library.rerank import re_ranking


def train_fn(dataloader, model, criterion, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        embeddings = model(images)
        loss = criterion(embeddings, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def extract_features(dataloader, model, device):
    """
    Extracts embeddings for the entire dataloader.
    Returns:
        features: torch.Tensor of shape (N, embedding_size)
        labels: np.array of shape (N,)
    """
    model.eval()
    features = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle cases where dataloader returns (images, labels) or just images
            if isinstance(batch, (tuple, list)):
                images = batch[0]
                if len(batch) > 1:
                    batch_labels = batch[1]
                    # Convert tensor labels to numpy if necessary
                    if isinstance(batch_labels, torch.Tensor):
                        batch_labels = batch_labels.cpu().numpy()
                    labels_list.extend(batch_labels)
            else:
                images = batch

            images = images.to(device)
            emb = model(images)
            features.append(emb.cpu())

    features = torch.cat(features, dim=0)

    if len(labels_list) > 0:
        return features, np.array(labels_list)
    else:
        return features, None


def eval_fn(query_loader, gallery_loader, model, device, label_encoder):
    """
    Evaluates the model using MAP@5 with the specified inference strategy:
    1. Global Retrieval (Cosine Similarity)
    2. Manifold Re-ranking
    3. Open-Set Rejection (Thresholding)
    """
    # 1. Extract Features
    query_feats, query_targets_enc = extract_features(query_loader, model, device)
    gallery_feats, gallery_targets_enc = extract_features(gallery_loader, model, device)

    # Decode targets (integers -> strings)
    # Note: query_targets_enc might contain -1 for 'new_whale' if handled in dataset
    # We reconstruct the string labels for ground truth
    query_labels = []
    for t in query_targets_enc:
        if t == -1:
            query_labels.append("new_whale")
        else:
            query_labels.append(label_encoder.inverse_transform([t])[0])

    gallery_labels = label_encoder.inverse_transform(gallery_targets_enc)

    # 2. Compute Cosine Similarity (for Open-Set Thresholding)
    # Normalize features
    q_norm = F.normalize(query_feats, p=2, dim=1)
    g_norm = F.normalize(gallery_feats, p=2, dim=1)

    # Compute Cosine Similarity Matrix (Query x Gallery)
    # Using torch.matmul for efficiency
    sim_matrix = torch.matmul(q_norm, g_norm.T)  # Shape: (N_query, N_gallery)

    # Get the maximum similarity for each query (nearest neighbor in cosine space)
    max_sim_vals, _ = torch.max(sim_matrix, dim=1)
    max_sim_vals = max_sim_vals.numpy()

    # 3. Compute Re-ranking Distances (for Ranking Known Candidates)
    # re_ranking expects numpy arrays and returns a distance matrix (smaller is better)
    dist_matrix = re_ranking(query_feats, gallery_feats)

    # 4. Generate Predictions
    final_preds = []

    for i in range(len(query_labels)):
        # Get distances for this query
        dists = dist_matrix[i]

        # Sort indices by distance (ascending) to get best candidates
        sorted_indices = np.argsort(dists)

        # Retrieve top 5 known candidates
        top_candidates = [gallery_labels[idx] for idx in sorted_indices[:5]]

        # Open-Set Logic:
        # If the similarity to the nearest known whale is below threshold, predict 'new_whale' first.
        if max_sim_vals[i] < Config.new_whale_threshold:
            # Primary prediction is new_whale, followed by best known guesses
            preds = ["new_whale"] + top_candidates[:4]
        else:
            # Predict known whales
            preds = top_candidates

        final_preds.append(preds)

    # 5. Calculate MAP@5
    score = map_at_5(final_preds, query_labels)

    return score


def generate_submission(test_loader, gallery_loader, model, device, label_encoder):
    """
    Generates predictions for the test set and saves them to submission.csv.
    """
    print("Generating submission...")

    # 1. Extract Features
    test_feats, _ = extract_features(test_loader, model, device)
    gallery_feats, gallery_targets_enc = extract_features(gallery_loader, model, device)

    gallery_labels = label_encoder.inverse_transform(gallery_targets_enc)

    # 2. Cosine Similarity (for Thresholding)
    q_norm = F.normalize(test_feats, p=2, dim=1)
    g_norm = F.normalize(gallery_feats, p=2, dim=1)
    sim_matrix = torch.matmul(q_norm, g_norm.T)
    max_sim_vals, _ = torch.max(sim_matrix, dim=1)
    max_sim_vals = max_sim_vals.numpy()

    # 3. Re-ranking
    dist_matrix = re_ranking(test_feats, gallery_feats)

    # 4. Generate Predictions
    results = []

    # Load test dataframe to get filenames
    test_df = pd.read_csv(Config.test_csv_path)
    test_filenames = test_df["Image"].values

    for i in range(len(test_filenames)):
        dists = dist_matrix[i]
        sorted_indices = np.argsort(dists)
        top_candidates = [gallery_labels[idx] for idx in sorted_indices[:5]]

        if max_sim_vals[i] < Config.new_whale_threshold:
            preds = ["new_whale"] + top_candidates[:4]
        else:
            preds = top_candidates

        pred_str = " ".join(preds)
        results.append({"Image": test_filenames[i], "Id": pred_str})

    submission_df = pd.DataFrame(results)
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")


def run_training(
    model,
    train_loader,
    val_loader,
    gallery_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    label_encoder,
    epochs=Config.epochs,
):
    """
    Main training loop with Early Stopping.
    """
    best_score = 0.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        # Train
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, device, scheduler
        )

        # Evaluate
        val_score = eval_fn(val_loader, gallery_loader, model, device, label_encoder)

        # Print Metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f} | Val MAP@5: {val_score:.10f}"
        )

        # Scheduler Step (ReduceLROnPlateau uses validation metric)
        if scheduler:
            # Assuming ReduceLROnPlateau, maximize=True for MAP@5
            # If the scheduler expects loss (minimize), we might need to invert or check config.
            # Standard ReduceLROnPlateau minimizes by default.
            # If mode='max', we pass score. If mode='min', we pass -score or loss.
            # We'll assume the scheduler is configured correctly in main.py,
            # but usually we pass the primary metric we want to improve.
            # Here we pass val_score assuming mode='max' or pass train_loss if mode='min'.
            # Given the ambiguity, passing val_score is safer for performance tracking if configured.
            # However, to be safe with default schedulers, we often step with loss.
            # Let's step with val_score and assume the user configured mode='max'.
            try:
                scheduler.step(val_score)
            except:
                scheduler.step()

        # Early Stopping & Checkpointing
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"New best model saved with MAP@5: {best_score:.10f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val MAP@5: {best_score:.10f}")

    # Load best model for inference
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    # Generate Submission
    generate_submission(test_loader, gallery_loader, model, device, label_encoder)
