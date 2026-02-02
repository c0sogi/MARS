import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import library.config as config
import library.utils as utils


class Trainer:
    """
    Manages the training lifecycle for the ArcFace Whale Identification model.
    """

    def __init__(self, model, train_loader, val_loader, gallery_loader, num_classes):
        """
        Args:
            model (nn.Module): The WhaleDenseNet model.
            train_loader (DataLoader): Loader for training data (with augmentation).
            val_loader (DataLoader): Loader for validation data.
            gallery_loader (DataLoader): Loader for the gallery (train set, no augmentation).
            num_classes (int): Number of unique known whale classes.
        """
        self.device = config.DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gallery_loader = gallery_loader
        self.num_classes = num_classes

        # Optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        # Cosine Annealing Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
        )

    def train_one_epoch(self, epoch_index):
        """
        Executes one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Passing labels triggers the ArcFace head to return logits
            logits = self.model(images, labels)

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        return epoch_loss

    def validate(self):
        """
        Evaluates the model using MAP@5 on the validation set.

        Strategy:
        1. Extract embeddings for the entire Gallery (Training set w/o aug).
        2. Extract embeddings for the Validation set.
        3. Compute Cosine Similarity Matrix.
        4. Retrieve Top-5 neighbors for each validation image.
        5. Compute MAP@5.
        """
        self.model.eval()

        # ---------------------------------------------------------
        # 1. Extract Gallery Embeddings
        # ---------------------------------------------------------
        gallery_embeddings = []
        gallery_labels = []

        with torch.no_grad():
            for images, labels, _ in self.gallery_loader:
                images = images.to(self.device)
                # No labels passed -> returns normalized embeddings
                embeds = self.model(images)
                gallery_embeddings.append(embeds.cpu())
                gallery_labels.append(labels)

        if not gallery_embeddings:
            return 0.0

        gallery_embeddings = torch.cat(gallery_embeddings)  # [N_gallery, Dim]
        gallery_labels = torch.cat(gallery_labels).numpy()  # [N_gallery]

        # ---------------------------------------------------------
        # 2. Extract Query (Validation) Embeddings
        # ---------------------------------------------------------
        query_embeddings = []
        query_labels = []

        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                embeds = self.model(images)
                query_embeddings.append(embeds.cpu())
                query_labels.append(labels)

        if not query_embeddings:
            return 0.0

        query_embeddings = torch.cat(query_embeddings)  # [N_query, Dim]
        query_labels = torch.cat(query_labels).numpy()  # [N_query]

        # ---------------------------------------------------------
        # 3. Compute Similarity & MAP@5
        # ---------------------------------------------------------
        # Move to GPU for fast matrix multiplication if possible
        # Check memory constraints; 7000x500 is small enough for GPU
        gal_emb_cuda = gallery_embeddings.to(self.device)
        qry_emb_cuda = query_embeddings.to(self.device)

        # Cosine Similarity = Dot product of normalized vectors
        # Shape: [N_query, N_gallery]
        sim_matrix = torch.matmul(qry_emb_cuda, gal_emb_cuda.t())

        # Get Top 5 indices
        # topk returns (values, indices)
        _, topk_indices = torch.topk(sim_matrix, k=5, dim=1)
        topk_indices = topk_indices.cpu().numpy()

        # Construct predictions list for MAP calculation
        predictions = []
        targets = []

        for i in range(len(query_labels)):
            true_label = query_labels[i]

            # Retrieve predicted labels from gallery indices
            pred_indices = topk_indices[i]
            pred_labels = [gallery_labels[idx] for idx in pred_indices]

            # Convert to strings for the map5 utility (or keep as ints)
            # utils.map5 expects lists of strings/objects, equality check works for ints too
            predictions.append(pred_labels)
            targets.append(true_label)

        score = utils.map5(predictions, targets)
        return score

    def fit(self, num_epochs=config.NUM_EPOCHS):
        """
        Runs the full training loop with Early Stopping.
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")

        best_map5 = 0.0
        patience = 5
        patience_counter = 0

        # Ensure working directory exists
        os.makedirs(config.WORKING_DIR, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Update LR
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Validate
            val_map5 = self.validate()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Loss: {train_loss:.6f} | "
                f"Val MAP@5: {val_map5:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpointing & Early Stopping
            if val_map5 > best_map5:
                best_map5 = val_map5
                patience_counter = 0
                torch.save(self.model.state_dict(), config.MODEL_PATH)
                print(f"  -> New Best Model Saved! (MAP@5: {best_map5:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered after {patience} epochs without improvement."
                    )
                    break

        print(f"Training complete. Best Validation MAP@5: {best_map5:.6f}")
        print(f"Best model saved to: {config.MODEL_PATH}")
