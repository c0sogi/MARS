import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from library.config import Config, seed_everything
from library.dataset import get_loaders
from library.model import WhaleModel
from library.loss import ArcFaceLoss
from library.utils import map_at_5


class Trainer:
    def __init__(self, debug=False):
        self.debug = debug
        self.device = Config.device
        self.best_score = 0.0
        seed_everything(Config.seed)

        # Create working directory if not exists
        os.makedirs(Config.working_dir, exist_ok=True)

    def get_optimizer(self, model, loss_module):
        # Combine parameters from backbone/head and the loss module (ArcFace centers)
        params = [
            {"params": model.parameters(), "lr": Config.learning_rate},
            {"params": loss_module.parameters(), "lr": Config.learning_rate},
        ]
        optimizer = optim.AdamW(
            params, lr=Config.learning_rate, weight_decay=Config.weight_decay
        )
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.scheduler_T_max, eta_min=Config.min_lr
        )
        return scheduler

    def train_fn(self, model, loss_fn, optimizer, scheduler, dataloader):
        model.train()
        loss_fn.train()

        running_loss = 0.0
        count = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()

            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        scheduler.step()
        epoch_loss = running_loss / count if count > 0 else 0.0
        return epoch_loss

    def extract_embeddings(self, model, dataloader):
        model.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                emb = model(images)
                embeddings.append(emb.cpu().numpy())

                if "label" in batch:
                    labels.append(batch["label"].numpy())

        embeddings = np.concatenate(embeddings)
        # Normalize embeddings for Cosine Similarity
        # (L2 Norm = 1, then Dot Product == Cosine Similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)

        if labels:
            labels = np.concatenate(labels)
            return embeddings, labels
        else:
            return embeddings, None

    def eval_fn(self, model, val_loader, gallery_loader, id2label):
        # Extract features
        val_embeddings, val_labels = self.extract_embeddings(model, val_loader)
        gallery_embeddings, gallery_labels = self.extract_embeddings(
            model, gallery_loader
        )

        # Create Label to ID string mapping
        # id2label is {str: int}, we need {int: str}
        # Note: 'new_whale' is mapped to -1 in dataset.py
        int2str = {v: k for k, v in id2label.items()}

        # Convert ground truth labels to strings
        ground_truth = [int2str[y] for y in val_labels]

        # Compute Similarity Matrix (Val x Gallery)
        # Shapes: (N_val, D) @ (N_gal, D).T -> (N_val, N_gal)
        # Using torch for matrix multiplication speed on GPU if possible, else numpy

        # Move to GPU for matrix calc if memory allows, otherwise CPU
        # Given 220GB RAM, CPU is safe. GPU 40GB might be tight for huge matrices but
        # (451 val) x (6789 gal) is very small.

        val_tensor = torch.from_numpy(val_embeddings).to(self.device)
        gal_tensor = torch.from_numpy(gallery_embeddings).to(self.device)

        sim_matrix = torch.matmul(val_tensor, gal_tensor.t())

        # Retrieve Top K neighbors
        # We need enough neighbors to find 5 unique IDs.
        # 20 is usually sufficient unless many neighbors have same ID.
        k = 50
        top_vals, top_indices = torch.topk(sim_matrix, k=k, dim=1)

        top_indices = top_indices.cpu().numpy()

        predictions = []
        for i in range(len(ground_truth)):
            indices = top_indices[i]

            # Map indices to Gallery IDs
            # gallery_labels is numpy array of ints
            neighbor_int_ids = gallery_labels[indices]

            # Get unique IDs preserving order
            unique_ids = []
            seen = set()
            for nid in neighbor_int_ids:
                if nid not in seen:
                    unique_ids.append(int2str[nid])
                    seen.add(nid)
                if len(unique_ids) == 5:
                    break

            predictions.append(unique_ids)

        score = map_at_5(predictions, ground_truth)

        # Cleanup
        del val_tensor, gal_tensor, sim_matrix
        gc.collect()

        return score

    def fit(self):
        print(f"Starting training on device: {self.device}")

        # 1. Load Data
        train_loader, gallery_loader, val_loader, _, id2label = get_loaders(
            debug=self.debug, load_cached_data=True
        )

        # 2. Setup Model & Loss
        model = WhaleModel(pretrained=True)
        model.to(self.device)

        # ArcFace Loss
        loss_fn = ArcFaceLoss()
        loss_fn.to(self.device)

        optimizer = self.get_optimizer(model, loss_fn)
        scheduler = self.get_scheduler(optimizer)

        # 3. Training Loop
        patience = 5
        patience_counter = 0

        for epoch in range(Config.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_fn(
                model, loss_fn, optimizer, scheduler, train_loader
            )

            # Validation
            val_score = self.eval_fn(model, val_loader, gallery_loader, id2label)

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{Config.epochs} - "
                f"Loss: {train_loss:.4f} - "
                f"Val MAP@5: {val_score:.10f} - "
                f"Time: {elapsed:.0f}s"
            )

            # Checkpoint & Early Stopping
            if val_score > self.best_score:
                self.best_score = val_score
                print(f"  New best score! Saving model to {Config.model_path}")
                torch.save(model.state_dict(), Config.model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

        print(f"Training finished. Best MAP@5: {self.best_score:.10f}")

        # Cleanup to free memory for inference
        del model, loss_fn, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
