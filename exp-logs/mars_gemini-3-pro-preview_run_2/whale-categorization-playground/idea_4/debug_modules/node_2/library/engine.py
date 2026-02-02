import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.loss import ArcFaceLoss
from library.utils import map_at_5
from library.rerank import re_ranking
from library.dataset import WhaleDataset, get_transforms


class WhaleEngine:
    def __init__(self, model, train_loader, val_loader, test_loader, label_encoder):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.label_encoder = label_encoder
        self.inverse_encoder = {v: k for k, v in label_encoder.items()}
        self.device = Config.DEVICE

        self.model.to(self.device)

        # Loss and Optimizer
        self.criterion = ArcFaceLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

        # Create a clean gallery loader (Training data without augmentation)
        self.gallery_loader = self._create_gallery_loader()

    def _create_gallery_loader(self):
        """
        Creates a DataLoader for the training set with Validation transforms (no augmentation).
        This serves as the Reference Gallery for validation and inference.
        """
        # Load metadata
        df = pd.read_csv(Config.TRAIN_CSV)
        # Filter new_whale (must match training data logic)
        df = df[df["Id"] != "new_whale"].reset_index(drop=True)

        # Determine cache path
        if Config.DEBUG:
            df = df.head(Config.DEBUG_SAMPLES)
            cache_path = Config.CACHE_TRAIN_IMAGES.replace(".npy", "_debug.npy")
        else:
            cache_path = Config.CACHE_TRAIN_IMAGES

        # Load images from cache
        if os.path.exists(cache_path):
            images = np.load(cache_path)
        else:
            # Fallback: This should ideally not happen if get_dataloaders ran first
            # We assume the cache exists. If not, we would need to recompute.
            # For simplicity in this engine module, we assume cache integrity.
            raise FileNotFoundError(f"Cache file {cache_path} not found.")

        # Prepare labels
        labels = df["Id"].map(self.label_encoder).values.astype(np.int64)

        # Create Dataset and Loader
        ds = WhaleDataset(images, labels, transform=get_transforms("val"))
        loader = DataLoader(
            ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        return loader

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        count = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (Training mode -> Logits)
            logits = self.model(images, labels)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        return running_loss / count if count > 0 else 0.0

    @torch.no_grad()
    def extract_features(self, loader):
        self.model.eval()
        feats = []
        labels_list = []

        for images, labels in loader:
            images = images.to(self.device)
            # Forward pass (Inference mode -> Embeddings)
            embeddings = self.model(images, labels=None)
            feats.append(embeddings.cpu())
            labels_list.append(labels)

        feats = torch.cat(feats, dim=0)
        labels_list = torch.cat(labels_list, dim=0)
        return feats, labels_list

    def valid_one_epoch(self):
        # Extract features for Query (Val) and Gallery (Clean Train)
        val_feats, val_labels = self.extract_features(self.val_loader)
        gallery_feats, gallery_labels = self.extract_features(self.gallery_loader)

        val_feats = val_feats.to(self.device)
        gallery_feats = gallery_feats.to(self.device)

        # Normalize
        val_feats = torch.nn.functional.normalize(val_feats, p=2, dim=1)
        gallery_feats = torch.nn.functional.normalize(gallery_feats, p=2, dim=1)

        # Compute Cosine Similarity (Query x Gallery)
        sim_matrix = torch.mm(val_feats, gallery_feats.t())

        # Get max similarity for thresholding (Open-Set Rejection)
        max_sim_vals, _ = torch.max(sim_matrix, dim=1)
        max_sim_vals = max_sim_vals.cpu().numpy()

        # Re-ranking or Standard Sorting
        if Config.USE_RERANKING:
            # re_ranking returns a distance matrix (smaller is better)
            dist_matrix = re_ranking(
                val_feats,
                gallery_feats,
                k1=Config.RERANK_K1,
                k2=Config.RERANK_K2,
                lambda_value=Config.RERANK_LAMBDA,
            )
            sorted_indices = np.argsort(dist_matrix, axis=1)
        else:
            # Sort by cosine similarity descending
            sorted_indices = (
                torch.argsort(sim_matrix, dim=1, descending=True).cpu().numpy()
            )

        # Generate Predictions
        val_preds = []
        val_gt = []

        gallery_labels_np = gallery_labels.numpy()
        val_labels_np = val_labels.numpy()

        for i in range(len(val_labels)):
            # Ground Truth
            gt_idx = val_labels_np[i]
            # Val set only contains known whales in this setup, but handle safely
            gt_id = self.inverse_encoder.get(gt_idx, "new_whale")
            val_gt.append(gt_id)

            # Retrieve Top Candidates
            top_indices = sorted_indices[i, :5]
            top_labels = gallery_labels_np[top_indices]
            top_ids = [self.inverse_encoder[lbl] for lbl in top_labels]

            # Apply Open-Set Logic
            if max_sim_vals[i] < Config.NEW_WHALE_THRESHOLD:
                # Predict new_whale first, then top 4 known
                pred_row = ["new_whale"] + top_ids[:4]
            else:
                pred_row = top_ids

            val_preds.append(pred_row)

        # Calculate Metric
        score = map_at_5(val_preds, val_gt)

        # Cleanup
        del val_feats, gallery_feats, sim_matrix
        if Config.USE_RERANKING:
            del dist_matrix
        gc.collect()

        return score

    def fit(self, num_epochs=Config.NUM_EPOCHS):
        best_score = 0.0
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_score = self.valid_one_epoch()

            # Update Scheduler
            self.scheduler.step(val_score)

            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val MAP@5: {val_score}"
            )

            # Checkpointing & Early Stopping
            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val MAP@5: {best_score}")

    def predict_test(self):
        print("Generating submission...")

        # Load Best Model
        if os.path.exists(Config.MODEL_SAVE_PATH):
            state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)

            # Cite debug_lesson_2: Handle Data-Dependent Layer Dimensions When Loading Checkpoints
            if "arc_head.weight" in state_dict:
                ckpt_shape = state_dict["arc_head.weight"].shape
                model_shape = self.model.arc_head.weight.shape
                if ckpt_shape != model_shape:
                    print(
                        f"Skipping arc_head.weight due to shape mismatch: {ckpt_shape} vs {model_shape}"
                    )
                    del state_dict["arc_head.weight"]

            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
        else:
            print("Warning: Best model not found. Using current weights.")

        # Extract Features
        test_feats, _ = self.extract_features(self.test_loader)
        gallery_feats, gallery_labels = self.extract_features(self.gallery_loader)

        test_feats = test_feats.to(self.device)
        gallery_feats = gallery_feats.to(self.device)

        # Normalize
        test_feats = torch.nn.functional.normalize(test_feats, p=2, dim=1)
        gallery_feats = torch.nn.functional.normalize(gallery_feats, p=2, dim=1)

        # Cosine Similarity
        sim_matrix = torch.mm(test_feats, gallery_feats.t())

        # Thresholding
        max_sim_vals, _ = torch.max(sim_matrix, dim=1)
        max_sim_vals = max_sim_vals.cpu().numpy()

        # Ranking
        if Config.USE_RERANKING:
            dist_matrix = re_ranking(
                test_feats,
                gallery_feats,
                k1=Config.RERANK_K1,
                k2=Config.RERANK_K2,
                lambda_value=Config.RERANK_LAMBDA,
            )
            sorted_indices = np.argsort(dist_matrix, axis=1)
        else:
            sorted_indices = (
                torch.argsort(sim_matrix, dim=1, descending=True).cpu().numpy()
            )

        # Format Submission
        test_images = pd.read_csv(Config.TEST_CSV)["Image"].values
        submission_data = []
        gallery_labels_np = gallery_labels.numpy()

        for i in range(len(test_images)):
            image_name = test_images[i]

            top_indices = sorted_indices[i, :5]
            top_labels = gallery_labels_np[top_indices]
            top_ids = [self.inverse_encoder[lbl] for lbl in top_labels]

            if max_sim_vals[i] < Config.NEW_WHALE_THRESHOLD:
                preds = ["new_whale"] + top_ids[:4]
            else:
                preds = top_ids

            pred_str = " ".join(preds)
            submission_data.append({"Image": image_name, "Id": pred_str})

        # Save
        df_sub = pd.DataFrame(submission_data)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
