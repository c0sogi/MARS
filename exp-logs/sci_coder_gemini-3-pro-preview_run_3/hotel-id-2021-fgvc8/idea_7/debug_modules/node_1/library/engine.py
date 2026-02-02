import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, accuracy, mapk


def train_fn(dataloader, model, criterion, optimizer, device, epoch):
    """
    Executes one epoch of training.
    """
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # ArcFace forward pass requires labels to compute margin loss
        outputs = model(images, labels)
        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping to stabilize training
        if Config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()

        # Compute accuracy on the logits
        acc = accuracy(outputs, labels, topk=(1,))[0]

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))

    print(f"Epoch [{epoch}] Train Loss: {loss_meter.avg} Accuracy: {acc_meter.avg}")
    return loss_meter.avg, acc_meter.avg


def inference_fn(dataloader, model, device):
    """
    Extracts features from the dataloader.
    Handles Test-Time Augmentation (TTA) if configured.
    Returns normalized features and labels (if available).
    """
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for data in dataloader:
            # Handle different dataloader outputs (image, label) vs (image)
            if isinstance(data, (list, tuple)):
                if len(data) == 2:
                    imgs, lbls = data
                    labels_list.append(lbls)
                else:
                    imgs = data[0]
            else:
                imgs = data

            imgs = imgs.to(device)

            # Extract features
            features = model.extract_features(imgs)

            # Test-Time Augmentation (Horizontal Flip)
            if Config.tta:
                imgs_flip = torch.flip(imgs, dims=[3])
                features_flip = model.extract_features(imgs_flip)
                features = (features + features_flip) / 2.0

            # Normalize features for Cosine Similarity
            features = F.normalize(features)
            features_list.append(features.cpu())

    features_all = torch.cat(features_list, dim=0)

    if len(labels_list) > 0:
        labels_all = torch.cat(labels_list, dim=0)
        return features_all, labels_all
    else:
        return features_all, None


def get_nearest_neighbors(features, model, device, k=5):
    """
    Computes the top-k nearest neighbors using Cosine Similarity
    between input features and the model's ArcFace class centers.
    """
    model.eval()
    features = features.to(device)

    # Get class centers (weights) from the ArcFace head
    # Weights shape: (num_classes, embedding_size)
    weights = model.arcface.weight.detach().to(device)

    # Normalize weights to ensure dot product equals cosine similarity
    weights = F.normalize(weights)

    # Compute Cosine Similarity
    # (N, Emb) @ (Emb, Classes) -> (N, Classes)
    similarity = torch.mm(features, weights.t())

    # Get top-k indices
    _, indices = torch.topk(similarity, k, dim=1)

    return indices.cpu().numpy()


def validate_fn(dataloader, model, device, unique_ids):
    """
    Runs validation and computes MAP@5.
    """
    print("Starting validation...")
    features, labels = inference_fn(dataloader, model, device)

    # Get predictions (indices)
    preds_indices = get_nearest_neighbors(features, model, device, k=5)

    # Prepare data for MAP@5 calculation
    # labels are indices (0..N-1), preds_indices are indices (0..N-1)
    # We can compare them directly.
    actual = [[l.item()] for l in labels]
    predicted = preds_indices.tolist()

    score = mapk(actual, predicted, k=5)

    # Print full precision as requested
    print(f"Validation MAP@5: {score}")
    return score


def generate_submission(dataloader, model, device, unique_ids):
    """
    Generates predictions for the test set and saves them to ./submission/submission.csv.
    """
    print("Generating submission...")

    # 1. Run inference
    features, _ = inference_fn(dataloader, model, device)

    # 2. Get Top-5 predictions (indices)
    preds_indices = get_nearest_neighbors(features, model, device, k=5)

    # 3. Map indices back to original Hotel IDs
    # unique_ids is an array where index i corresponds to the hotel_id at that index
    final_preds = []
    for row_indices in preds_indices:
        # Map each index in the top-5 to its hotel_id
        mapped_ids = [str(unique_ids[idx]) for idx in row_indices]
        final_preds.append(" ".join(mapped_ids))

    # 4. Create DataFrame
    # Load test metadata to ensure image order matches
    test_df = pd.read_csv(Config.test_metadata_path)

    submission_df = pd.DataFrame({"image": test_df["image"], "hotel_id": final_preds})

    # 5. Save submission
    output_dir = "./submission"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "submission.csv")

    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
