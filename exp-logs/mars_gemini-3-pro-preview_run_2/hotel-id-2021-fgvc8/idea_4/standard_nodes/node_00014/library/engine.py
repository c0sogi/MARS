import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gc
import time
from library.config import CFG
from library.utils import AverageMeter, map_at_k


def train_fn(dataloader, model, criterion, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.
    Computes Multi-Task Loss: HotelID (SubCenter) + ChainID (ArcFace).
    """
    model.train()

    loss_meter = AverageMeter()
    hotel_loss_meter = AverageMeter()
    chain_loss_meter = AverageMeter()

    # Iterate over dataloader
    # Note: Progress bar is suppressed as per requirements
    for step, (images, hotel_labels, chain_labels) in enumerate(dataloader):
        images = images.to(device)
        hotel_labels = hotel_labels.to(device)
        chain_labels = chain_labels.to(device)

        batch_size = images.size(0)

        # Forward pass
        # model.forward() in train mode returns (hotel_logits, chain_logits)
        hotel_logits, chain_logits = model(images, hotel_labels, chain_labels)

        # 1. Hotel Loss (Fine-Grained)
        # SubCenterArcFace outputs logits of shape (Batch, Num_Classes) after max-pooling K centers
        loss_hotel = criterion(hotel_logits, hotel_labels)

        # 2. Chain Loss (Coarse-Grained)
        # We must mask out samples where chain_id == 0 (unknown/no chain)
        # ArcFace outputs logits of shape (Batch, Num_Chains)
        mask = chain_labels != 0
        if mask.sum() > 0:
            loss_chain = criterion(chain_logits[mask], chain_labels[mask])
        else:
            loss_chain = torch.tensor(0.0, device=device)

        # Total Loss
        loss = loss_hotel + (CFG.lambda_chain * loss_chain)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Update meters
        loss_meter.update(loss.item(), batch_size)
        hotel_loss_meter.update(loss_hotel.item(), batch_size)
        chain_loss_meter.update(loss_chain.item(), batch_size)

        if CFG.debug and step >= 10:
            break

    return loss_meter.avg, hotel_loss_meter.avg, chain_loss_meter.avg


def valid_fn(dataloader, model, criterion, device):
    """
    Performs validation.
    Calculates MAP@5 using the Hotel Head's predictions.
    """
    model.eval()

    loss_meter = AverageMeter()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for step, (images, hotel_labels, chain_labels) in enumerate(dataloader):
            images = images.to(device)
            hotel_labels = hotel_labels.to(device)
            chain_labels = chain_labels.to(device)
            batch_size = images.size(0)

            # Extract features manually to pass through heads
            # model.forward() in eval mode returns embeddings only
            features = model.extract_features(images)

            # Calculate Logits for Loss and Metrics
            # We must pass labels to heads to calculate loss (ArcFace requirement),
            # but for prediction ranking we use the logits.
            hotel_logits = model.hotel_head(features, hotel_labels)

            # Chain logits for loss calculation
            chain_logits = model.chain_head(features, chain_labels)

            # Compute Loss
            loss_hotel = criterion(hotel_logits, hotel_labels)

            mask = chain_labels != 0
            if mask.sum() > 0:
                loss_chain = criterion(chain_logits[mask], chain_labels[mask])
            else:
                loss_chain = torch.tensor(0.0, device=device)

            loss = loss_hotel + (CFG.lambda_chain * loss_chain)
            loss_meter.update(loss.item(), batch_size)

            # Predictions for MAP@5
            # hotel_logits is (Batch, Num_Classes)
            # We take top 5 indices
            _, top_k_indices = torch.topk(hotel_logits, k=5, dim=1)

            preds_list.append(top_k_indices.cpu().numpy())
            targets_list.append(hotel_labels.cpu().numpy())

            if CFG.debug and step >= 10:
                break

    # Concatenate all batches
    predictions = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # Calculate MAP@5
    map_score = map_at_k(predictions, targets, k=5)

    return loss_meter.avg, map_score


def train_loop(train_loader, val_loader, model, optimizer, scheduler, device, epochs):
    """
    Manages the full training loop including Early Stopping.
    """
    criterion = nn.CrossEntropyLoss()
    best_map = 0.0
    patience = 3
    patience_counter = 0
    best_model_path = os.path.join(CFG.working_dir, "best_model.pth")
    saved_any_model = False

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss, train_hotel_loss, train_chain_loss = train_fn(
            train_loader, model, criterion, optimizer, device, scheduler
        )

        # Validate
        val_loss, val_map = valid_fn(val_loader, model, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} [{elapsed:.0f}s]: "
            f"Train Loss: {train_loss:.6f} (Hotel: {train_hotel_loss:.6f}, Chain: {train_chain_loss:.6f}) | "
            f"Val Loss: {val_loss:.6f} | Val MAP@5: {val_map:.6f}"
        )

        # Checkpointing
        if val_map > best_map:
            best_map = val_map
            print(
                f"  -> MAP improved from {best_map:.6f} to {val_map:.6f}. Saving model..."
            )
            torch.save(model.state_dict(), best_model_path)
            saved_any_model = True
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> MAP did not improve. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model for final state
    if saved_any_model and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model.")

    return model


def inference_fn(test_loader, model, device, hotel_classes):
    """
    Generates predictions for the test set.
    Implements Test-Time Augmentation (TTA) and Query Expansion (QE).
    """
    model.eval()

    embeddings = []
    image_names = []

    print("Extracting features for test set...")
    with torch.no_grad():
        for step, (images, names) in enumerate(test_loader):
            images = images.to(device)

            # 1. TTA: Original + Horizontal Flip
            if CFG.use_tta:
                images_flipped = torch.flip(images, dims=[3])  # N, C, H, W -> flip W

                emb_orig = model(images)  # Returns embeddings in eval mode
                emb_flip = model(images_flipped)

                # Average and normalize
                emb = (emb_orig + emb_flip) / 2.0
            else:
                emb = model(images)

            emb = F.normalize(emb, p=2, dim=1)

            embeddings.append(emb.cpu())
            image_names.extend(names)

            if CFG.debug and step >= 5:
                break

    embeddings = torch.cat(embeddings, dim=0).to(device)  # (N_test, Dim)

    # 2. Query Expansion (QE)
    # We use the Class Centers (Weights of Hotel Head) as the Gallery.
    # SubCenterArcFace weights are (Num_Classes * K, Dim).
    # We retrieve top neighbors from these centers to refine the query.
    if CFG.use_qe:
        print(f"Applying Query Expansion (k={CFG.qe_k})...")

        # Get class centers
        gallery_weights = model.hotel_head.weight  # (C*K, D)
        gallery_weights = F.normalize(gallery_weights, p=2, dim=1)

        # Compute similarity between Queries and Gallery (Class Centers)
        # sim: (N_test, C*K)
        sim = torch.matmul(embeddings, gallery_weights.T)

        # Retrieve top K neighbors
        _, top_idxs = torch.topk(sim, k=CFG.qe_k, dim=1)  # (N_test, k)

        # Gather neighbor vectors
        # neighbors: (N_test, k, D)
        neighbors = gallery_weights[top_idxs]

        # Average Query Expansion: New Query = (Old Query + Sum(Neighbors)) / (1 + k)
        # Note: We can also do weighted average, but simple average is standard for AQE
        embeddings_expanded = (embeddings + neighbors.sum(dim=1)) / (1.0 + CFG.qe_k)
        embeddings_expanded = F.normalize(embeddings_expanded, p=2, dim=1)

        # Update embeddings for final classification
        final_embeddings = embeddings_expanded
    else:
        final_embeddings = embeddings

    # 3. Final Classification
    # We compute dot product with class centers again to get scores
    print("Generating final predictions...")
    gallery_weights = model.hotel_head.weight
    gallery_weights = F.normalize(gallery_weights, p=2, dim=1)

    # Sim: (N_test, C*K)
    sim = torch.matmul(final_embeddings, gallery_weights.T)

    # Handle SubCenter: Reshape to (N, C, K) and take max over K
    sim = sim.view(sim.size(0), CFG.num_classes, CFG.subcenter_k)
    scores, _ = torch.max(sim, dim=2)  # (N, C)

    # Get top 5 predictions
    _, top5_indices = torch.topk(scores, k=5, dim=1)
    top5_indices = top5_indices.cpu().numpy()

    # Map indices back to original Hotel IDs
    final_preds = []
    for idx_list in top5_indices:
        pred_hotels = [str(hotel_classes[i]) for i in idx_list]
        final_preds.append(" ".join(pred_hotels))

    # Create submission DataFrame
    submission_df = pd.DataFrame({"image": image_names, "hotel_id": final_preds})

    # Save submission
    sub_path = os.path.join("submission", "submission.csv")
    os.makedirs("submission", exist_ok=True)
    submission_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    return submission_df
