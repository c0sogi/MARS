import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import AverageMeter, seed_everything
from library.dataset import HotelDataset, get_transforms, get_label_mapping


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_fn(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    loss_meter = AverageMeter()

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def valid_fn(model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, labels)
            loss = criterion(outputs, labels)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1.item(), images.size(0))
            acc5_meter.update(acc5.item(), images.size(0))

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


def generate_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle both (image, label) and (image, image_name)
            if len(batch) == 2:
                images, targets = batch
            else:
                images = batch[0]
                targets = None

            images = images.to(device)

            # Get embeddings (pass labels=None to get features from neck)
            emb = model(images, labels=None)

            # Normalize embeddings for Cosine Similarity
            emb = F.normalize(emb, p=2, dim=1)

            embeddings.append(emb.cpu().numpy())

            if targets is not None:
                if isinstance(targets, torch.Tensor):
                    labels.append(targets.cpu().numpy())
                else:
                    labels.extend(targets)

    embeddings = np.concatenate(embeddings)
    if len(labels) > 0 and isinstance(labels[0], np.ndarray):
        labels = np.concatenate(labels)

    return embeddings, labels


def run_training(
    model, train_loader, val_loader, num_epochs=Config.EPOCHS, device=Config.DEVICE
):
    seed_everything(Config.SEED)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_MIN_LR
    )

    best_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Warmup Logic
        if epoch < Config.WARMUP_EPOCHS:
            print(f"Epoch {epoch+1}: Warmup Mode - Freezing Backbone")
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        else:
            if epoch == Config.WARMUP_EPOCHS:
                print(f"Epoch {epoch+1}: Fine-tuning Mode - Unfreezing Backbone")
            for param in model.backbone.parameters():
                param.requires_grad = True

        train_loss = train_fn(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc1, val_acc5 = valid_fn(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {elapsed:.0f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Acc@1: {val_acc1}")
        print(f"Val Acc@5: {val_acc5}")

        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"Saved Best Model (Loss: {best_loss})")

    # Save final model
    torch.save(model.state_dict(), Config.MODEL_PATH)


def predict_and_submit(model, test_loader, device=Config.DEVICE, load_cached_data=True):
    model.to(device)
    model.eval()

    # --- 1. Gallery Generation (Train Set) ---
    if (
        load_cached_data
        and os.path.exists(Config.GALLERY_EMB_PATH)
        and os.path.exists(Config.GALLERY_LABELS_PATH)
    ):
        print(f"Loading cached gallery embeddings from {Config.GALLERY_EMB_PATH}")
        gallery_emb = np.load(Config.GALLERY_EMB_PATH)
        gallery_labels = np.load(Config.GALLERY_LABELS_PATH)
    else:
        print("Generating gallery embeddings...")
        # Create a clean loader for the training set (no augmentations, no shuffle, no drop_last)
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

        # We need label mapping to ensure correct labels
        id_to_idx, idx_to_id = get_label_mapping(load_cached_data=True)
        train_df["label_idx"] = train_df["hotel_id"].map(id_to_idx).astype(int)

        gallery_dataset = HotelDataset(
            train_df, transform=get_transforms(mode="val"), mode="train"
        )
        gallery_loader = DataLoader(
            gallery_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        gallery_emb, gallery_labels = generate_embeddings(model, gallery_loader, device)

        # Save to cache
        np.save(Config.GALLERY_EMB_PATH, gallery_emb)
        np.save(Config.GALLERY_LABELS_PATH, gallery_labels)
        print("Gallery embeddings saved.")

    # --- 2. Query Generation (Test Set) ---
    if load_cached_data and os.path.exists(Config.QUERY_EMB_PATH):
        print(f"Loading cached query embeddings from {Config.QUERY_EMB_PATH}")
        query_emb = np.load(Config.QUERY_EMB_PATH)
        # Load test metadata to get image names
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        query_names = test_df["image"].values
    else:
        print("Generating query embeddings...")
        query_emb, query_names = generate_embeddings(model, test_loader, device)
        np.save(Config.QUERY_EMB_PATH, query_emb)
        print("Query embeddings saved.")

    # --- 3. Retrieval (KNN) ---
    print("Running retrieval...")
    gallery_tensor = torch.from_numpy(gallery_emb).to(device)  # (N_gal, Dim)
    query_tensor = torch.from_numpy(query_emb).to(device)  # (N_query, Dim)

    # Cosine Similarity: sim = Q @ G.T
    sim_matrix = torch.matmul(query_tensor, gallery_tensor.t())  # (N_query, N_gal)

    # Get Top K
    topk_vals, topk_indices = torch.topk(sim_matrix, k=Config.KNN_K, dim=1)

    topk_indices = topk_indices.cpu().numpy()

    # --- 4. Format Submission ---
    print("Formatting submission...")
    _, idx_to_id = get_label_mapping(load_cached_data=True)

    submission_rows = []

    for i, q_name in enumerate(query_names):
        indices = topk_indices[i]

        # Map indices to hotel IDs
        retrieved_label_indices = gallery_labels[indices]

        # Get 5 unique hotels, preserving order
        unique_hotels = []
        seen = set()
        for label_idx in retrieved_label_indices:
            if label_idx not in seen:
                hotel_id = idx_to_id[label_idx]
                unique_hotels.append(str(hotel_id))
                seen.add(label_idx)
                if len(unique_hotels) == 5:
                    break

        prediction_str = " ".join(unique_hotels)
        submission_rows.append({"image": q_name, "hotel_id": prediction_str})

    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
