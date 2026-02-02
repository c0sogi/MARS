import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config, seed_everything
from library.utils import AverageMeter, mapk
from library.dataset import HotelDataset, get_transforms, get_class_mapping
from library.model import HotelIdModel


def load_class_mapping(load_cached_data=True):
    """
    Loads the hotel_id to class_idx mapping.
    Implements caching using Parquet as required.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "class_mapping.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Convert to dictionary: hotel_id -> class_idx
            mapping = dict(zip(df["hotel_id"], df["class_idx"]))
            return mapping
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute mapping
    mapping = get_class_mapping(Config.TRAIN_META)

    # Cache result
    try:
        df = pd.DataFrame(list(mapping.items()), columns=["hotel_id", "class_idx"])
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return mapping


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle for the Hotel ID task.
    Implements the Plasticity-Preserving Two-Stage Curriculum.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.device = Config.DEVICE

        # Data Mappings
        self.class_to_idx = load_class_mapping(load_cached_data=True)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Model
        self.model = HotelIdModel(
            n_classes=len(self.class_to_idx),
            backbone_name=Config.BACKBONE,
            embedding_dim=Config.EMBEDDING_DIM,
            pretrained=Config.PRETRAINED,
            gem_p=Config.GEM_P,
            margin=Config.MARGIN,
            scale=Config.SCALE,
        )
        self.model.to(self.device)

        # Optimization
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=Config.TOTAL_EPOCHS, eta_min=Config.MIN_LR
        )

        self.criterion = nn.CrossEntropyLoss()

    def get_dataloader(self, mode):
        """Creates DataLoaders for train, val, or test."""
        if mode == "train":
            csv_path = Config.TRAIN_META
            shuffle = True
            transform = get_transforms("train")
        elif mode == "val":
            csv_path = Config.VAL_META
            shuffle = False
            transform = get_transforms("val")
        elif mode == "test":
            csv_path = Config.TEST_META
            shuffle = False
            transform = get_transforms("test")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        dataset = HotelDataset(
            csv_path=csv_path,
            transform=transform,
            class_to_idx=self.class_to_idx,
            mode=mode,
        )

        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=(mode == "train"),
        )

    def train_one_epoch(self, epoch, train_loader):
        """
        Executes one training epoch.
        Handles the curriculum logic for ArcFace margin.
        """
        self.model.train()
        loss_meter = AverageMeter()

        # Curriculum: Adjust Margin
        # Stage 1: Warmup (Softmax, m=0)
        # Stage 2: Metric Learning (ArcFace, m=Config.MARGIN)
        if epoch < Config.WARMUP_EPOCHS:
            current_margin = 0.0
        else:
            current_margin = Config.MARGIN

        # Update model head parameters
        self.model.head.m = current_margin
        # Ensure scale is set (fixed at 30.0 as per plan)
        self.model.head.s = Config.SCALE

        for i, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with labels -> returns logits (with margin applied)
            outputs = self.model(images, labels)

            loss = self.criterion(outputs, labels)
            loss.backward()

            self.optimizer.step()

            loss_meter.update(loss.item(), images.size(0))

        return loss_meter.avg

    def valid_one_epoch(self, val_loader):
        """
        Executes validation.
        Computes MAP@5 using embeddings and class prototypes.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            # Pre-compute normalized class centers (prototypes)
            # weight shape: (out_features, in_features)
            prototypes = F.normalize(self.model.head.weight, p=2, dim=1)

            for batch in val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass without labels -> returns normalized embeddings
                embeddings = self.model(images, labels=None)

                # Compute cosine similarity logits: (B, Embed) @ (Embed, Classes) -> (B, Classes)
                # We use the raw cosine similarity for ranking (scale s doesn't change order)
                logits = torch.matmul(embeddings, prototypes.t())

                # Get top 5 predictions
                _, top_k_indices = torch.topk(logits, k=5, dim=1)

                all_preds.extend(top_k_indices.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())

        # Calculate MAP@5
        # targets is a list of scalars, preds is a list of lists
        val_map = mapk(all_targets, all_preds, k=5)
        return val_map

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")
        print(
            f"Curriculum: {Config.WARMUP_EPOCHS} epochs warmup (m=0), then ArcFace (m={Config.MARGIN})"
        )

        train_loader = self.get_dataloader("train")
        val_loader = self.get_dataloader("val")

        best_map = 0.0
        patience = 4
        patience_counter = 0

        for epoch in range(Config.TOTAL_EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch, train_loader)

            # Validation
            val_map = self.valid_one_epoch(val_loader)

            # Scheduler Step
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Logging (Full precision as requested)
            print(
                f"Epoch {epoch+1}/{Config.TOTAL_EPOCHS} | "
                f"Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Val MAP@5: {val_map}"
            )

            # Checkpointing & Early Stopping
            if val_map > best_map:
                best_map = val_map
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"New best model saved with MAP@5: {best_map}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print("Training complete.")

        # Generate Submission
        self.predict_and_submit()

    def predict_and_submit(self):
        """
        Loads the best model, predicts on test set, and generates submission.csv.
        """
        print("Generating submission...")

        # Load best model
        if os.path.exists(Config.MODEL_PATH):
            state_dict = torch.load(Config.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Loaded best model weights.")
        else:
            print("Warning: Best model not found. Using current weights.")

        self.model.eval()
        test_loader = self.get_dataloader("test")

        results = []

        with torch.no_grad():
            prototypes = F.normalize(self.model.head.weight, p=2, dim=1)

            for batch in test_loader:
                images = batch["image"].to(self.device)
                image_ids = batch["image_id"]  # List of strings

                embeddings = self.model(images, labels=None)
                logits = torch.matmul(embeddings, prototypes.t())
                _, top_k_indices = torch.topk(logits, k=5, dim=1)

                top_k_indices = top_k_indices.cpu().numpy()

                for img_id, indices in zip(image_ids, top_k_indices):
                    # Map indices back to hotel_ids
                    hotel_ids = [str(self.idx_to_class[idx]) for idx in indices]
                    prediction_str = " ".join(hotel_ids)
                    results.append({"image": img_id, "hotel_id": prediction_str})

        # Save to CSV
        submission_df = pd.DataFrame(results)
        # Ensure column order
        submission_df = submission_df[["image", "hotel_id"]]
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
