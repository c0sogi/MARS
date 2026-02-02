import os
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from library.config import Config
from library.utils import compute_map5, save_submission
from library.model import WhaleModel
from library.loss import AdaFaceLoss


class Trainer:
    """
    Manages training, validation, and inference for the Whale Identification task.
    Implements EfficientNet + AdaFace training with Retrieval-based Validation
    and Query Expansion Inference.
    """

    def __init__(self, train_loader, gallery_loader, val_loader, encoder):
        """
        Args:
            train_loader: DataLoader for training (augmented).
            gallery_loader: DataLoader for gallery (train set, no aug).
            val_loader: DataLoader for validation (query set).
            encoder: Fitted LabelEncoder to map integers back to strings.
        """
        self.train_loader = train_loader
        self.gallery_loader = gallery_loader
        self.val_loader = val_loader
        self.encoder = encoder
        self.device = Config.device
        self.classes = encoder.classes_
        self.num_classes = len(self.classes)

        # Initialize Model
        self.model = WhaleModel(num_classes=self.num_classes).to(self.device)

        # Loss Function
        self.criterion = AdaFaceLoss().to(self.device)

        # Optimizer (AdamW)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

        # Scheduler (ReduceLROnPlateau based on MAP@5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=Config.scheduler_patience,
        )

        # State tracking
        self.best_score = 0.0
        self.early_stop_counter = 0

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with targets for AdaFace margin calculation
            logits = self.model(images, targets=labels)

            loss = self.criterion(logits, labels)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        epoch_loss = running_loss / count if count > 0 else 0.0
        return epoch_loss

    def extract_embeddings(self, loader):
        """
        Extracts normalized embeddings for a given loader.
        Returns:
            embeddings (np.array): Shape (N, embedding_dim)
            targets (np.array): Shape (N,) - Labels or Image Names
        """
        self.model.eval()
        embeddings = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                # Unpack batch (could be image, label OR image, image_name)
                images = batch[0]
                labels = batch[1]

                images = images.to(self.device)

                # Forward without targets returns normalized embeddings (L2=1)
                emb = self.model(images)

                embeddings.append(emb.cpu().numpy())

                # Handle labels (tensor) or names (tuple/list)
                if isinstance(labels, torch.Tensor):
                    targets.append(labels.numpy())
                else:
                    targets.extend(labels)

        if len(embeddings) > 0:
            embeddings = np.vstack(embeddings)

            if isinstance(targets[0], np.ndarray) or isinstance(
                targets[0], (int, float, np.integer)
            ):
                targets = np.concatenate(targets)
            else:
                targets = np.array(targets)
        else:
            embeddings = np.array([])
            targets = np.array([])

        return embeddings, targets

    def validate(self):
        """
        Performs Retrieval-based Validation.
        1. Extract Gallery (Train) and Query (Val) embeddings.
        2. KNN Search.
        3. Compute MAP@5.
        """
        # Extract embeddings
        gallery_emb, gallery_labels = self.extract_embeddings(self.gallery_loader)
        query_emb, query_labels = self.extract_embeddings(self.val_loader)

        if len(gallery_emb) == 0 or len(query_emb) == 0:
            return 0.0

        # KNN Search (Cosine Similarity)
        # Since embeddings are normalized, we can use Euclidean distance for speed/stability
        # or Cosine. Sklearn's cosine is 1 - cos_sim.
        knn = NearestNeighbors(n_neighbors=Config.knn_k, metric="cosine", n_jobs=-1)
        knn.fit(gallery_emb)

        # Find neighbors
        _, indices = knn.kneighbors(query_emb)

        # Retrieve predicted labels from gallery
        # indices shape: (num_queries, k)
        predicted_labels = gallery_labels[indices]

        # Compute MAP@5
        score = compute_map5(query_labels, predicted_labels)
        return score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on {self.device}...")

        for epoch in range(1, Config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_score = self.validate()

            # Scheduler step
            self.scheduler.step(val_score)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{Config.epochs} - Loss: {train_loss} - Val MAP@5: {val_score} - LR: {current_lr}"
            )

            # Save Best Model
            if val_score > self.best_score:
                self.best_score = val_score
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), Config.model_save_path)
            else:
                self.early_stop_counter += 1

            # Early Stopping
            if self.early_stop_counter >= Config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print(f"Training complete. Best Val MAP@5: {self.best_score}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set using Query Expansion and Thresholding.
        Saves submission file.
        """
        print("Starting Inference...")

        # Load best model
        if os.path.exists(Config.model_save_path):
            self.model.load_state_dict(
                torch.load(Config.model_save_path, map_location=self.device)
            )
            print("Loaded best model checkpoint.")

        # 1. Extract Embeddings
        # Gallery: Train set (Known Whales)
        gallery_emb, gallery_labels = self.extract_embeddings(self.gallery_loader)
        # Query: Test set
        test_emb, test_names = self.extract_embeddings(test_loader)

        # 2. Initial KNN for Query Expansion
        knn = NearestNeighbors(n_neighbors=Config.knn_k, metric="cosine", n_jobs=-1)
        knn.fit(gallery_emb)

        # Initial search
        dists, indices = knn.kneighbors(test_emb)

        final_preds = []

        # 3. Query Expansion & Thresholding
        for i in range(len(test_emb)):
            # Query Expansion
            if Config.use_qe:
                # Get indices of top neighbors for expansion
                # Use top qe_k neighbors
                neighbor_indices = indices[i, : Config.qe_k]
                neighbor_embs = gallery_emb[neighbor_indices]

                # Average original query + neighbors
                # (Simple average, could be weighted by similarity)
                expanded_query = test_emb[i] + np.mean(neighbor_embs, axis=0)

                # Re-normalize
                norm = np.linalg.norm(expanded_query)
                expanded_query = expanded_query / (norm + 1e-6)

                # Reshape for single query
                query_vec = expanded_query.reshape(1, -1)
            else:
                query_vec = test_emb[i].reshape(1, -1)

            # Re-query with expanded vector
            # We need distances for thresholding
            q_dists, q_indices = knn.kneighbors(query_vec, n_neighbors=5)

            # Get top matches
            top_indices = q_indices[0]
            top_dists = q_dists[0]

            # Map to Class IDs (integers)
            top_class_ints = gallery_labels[top_indices]
            # Convert to Strings
            top_class_strs = self.encoder.inverse_transform(top_class_ints)

            # Thresholding Logic
            # Cosine distance = 1 - similarity
            # If similarity < threshold => distance > (1 - threshold)
            # If top match is weak, predict 'new_whale' first

            top_dist = top_dists[0]
            top_sim = 1.0 - top_dist

            current_preds = []

            if top_sim < Config.unknown_threshold:
                current_preds.append("new_whale")
                # Append remaining top predictions until we have 5
                for label in top_class_strs:
                    if len(current_preds) < 5:
                        current_preds.append(label)
            else:
                # Confident match
                current_preds.extend(list(top_class_strs))
                # Ensure 'new_whale' is not in the list if we are confident?
                # Or append it at the end if space permits?
                # The task requires 5 labels. If we fill 5 with knowns, we are done.
                # If we have duplicates (unlikely with unique gallery indices but possible if gallery has same labels),
                # we should handle uniqueness.

                # Deduplicate while preserving order
                seen = set()
                unique_preds = []
                for p in current_preds:
                    if p not in seen:
                        unique_preds.append(p)
                        seen.add(p)

                # Fill with 'new_whale' if space
                if len(unique_preds) < 5:
                    unique_preds.append("new_whale")

                current_preds = unique_preds[:5]

            final_preds.append(current_preds)

        # 4. Save Submission
        save_submission(test_names, final_preds, Config.submission_path)
        print(f"Submission saved to {Config.submission_path}")
