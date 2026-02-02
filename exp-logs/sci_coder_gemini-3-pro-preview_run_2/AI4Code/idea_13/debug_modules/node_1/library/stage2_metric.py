import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything
from library.data_loader import NotebookProcessor, MetricLearningDataset
from library.feature_engineering import TextVectorizer


class SiameseProjector(nn.Module):
    """
    A lightweight MLP to project SVD embeddings into a metric space
    optimized for ranking tasks.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(SiameseProjector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        return self.net(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        # Contrastive loss formula
        # label is 1 for positive (similar), 0 for negative (dissimilar)
        # Note: The standard formula often uses Y=1 for similar.
        # Loss = Y * d^2 + (1-Y) * max(0, margin - d)^2

        loss_contrastive = torch.mean(
            (label * torch.pow(euclidean_distance, 2))
            + (
                (1 - label)
                * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
            )
        )

        return loss_contrastive


class Stage2Metric:
    """
    Manager class for Stage 2: Supervised Metric Learning.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vectorizer = TextVectorizer(config)

    def train(self):
        """
        Trains the Siamese Projector model.
        """
        seed_everything(self.config.SEED)
        print(f"Running Stage 2 Training on device: {self.device}")

        # 1. Load Data
        print("Loading Dataframes...")
        processor = NotebookProcessor(self.config)
        df_train = processor.load_data("train")
        df_val = processor.load_data("val")

        # 2. Vectorization (SVD)
        # We need dense SVD features for the neural network
        print("Generating SVD features...")
        train_texts = df_train["source"].fillna("").astype(str).tolist()
        val_texts = df_val["source"].fillna("").astype(str).tolist()

        # Fit on train, transform both
        self.vectorizer.fit(train_texts, load_cached_models=True)
        X_train = self.vectorizer.transform(train_texts)
        X_val = self.vectorizer.transform(val_texts)

        # 3. Create Datasets
        print("Creating Metric Learning Datasets...")
        train_dataset = MetricLearningDataset(df_train, X_train, mode="train")
        val_dataset = MetricLearningDataset(df_val, X_val, mode="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.METRIC_BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        # Validation loader can be larger as no gradients are stored
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.METRIC_BATCH_SIZE * 2,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        # 4. Initialize Model
        model = SiameseProjector(
            input_dim=self.config.METRIC_INPUT_DIM,
            hidden_dim=self.config.METRIC_HIDDEN_DIM,
            output_dim=self.config.METRIC_EMBEDDING_DIM,
            dropout=self.config.METRIC_DROPOUT,
        ).to(self.device)

        criterion = ContrastiveLoss(margin=self.config.METRIC_MARGIN)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.METRIC_LR,
            weight_decay=self.config.METRIC_WEIGHT_DECAY,
        )

        # 5. Training Loop
        best_val_loss = float("inf")
        patience_counter = 0

        print("Starting Training Loop...")
        for epoch in range(self.config.METRIC_EPOCHS):
            model.train()
            running_loss = 0.0

            # Train Step
            # Using tqdm for progress tracking within epoch (optional, but requested silent mostly)
            # We will just iterate
            for x1, x2, label in train_loader:
                x1, x2, label = (
                    x1.to(self.device),
                    x2.to(self.device),
                    label.to(self.device),
                )

                optimizer.zero_grad()
                out1 = model(x1)
                out2 = model(x2)

                loss = criterion(out1, out2, label.unsqueeze(1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x1.size(0)

            epoch_train_loss = running_loss / len(train_dataset)

            # Validation Step
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for x1, x2, label in val_loader:
                    x1, x2, label = (
                        x1.to(self.device),
                        x2.to(self.device),
                        label.to(self.device),
                    )
                    out1 = model(x1)
                    out2 = model(x2)
                    loss = criterion(out1, out2, label.unsqueeze(1))
                    val_running_loss += loss.item() * x1.size(0)

            epoch_val_loss = val_running_loss / len(val_dataset)

            print(
                f"Epoch {epoch+1}/{self.config.METRIC_EPOCHS} - Train Loss: {epoch_train_loss} - Val Loss: {epoch_val_loss}"
            )

            # Checkpoint & Early Stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), self.config.METRIC_MODEL_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:  # Hardcoded patience for simplicity/speed
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best model saved to {self.config.METRIC_MODEL_PATH}")

    def get_projected_embeddings(self, texts, load_cached_model=True):
        """
        Generates projected embeddings for a list of texts using the trained model.

        Args:
            texts (list): List of text strings.
            load_cached_model (bool): Whether to load the saved model weights.

        Returns:
            np.ndarray: Projected embeddings (N, embedding_dim).
        """
        # 1. Vectorize (SVD)
        # Ensure vectorizer is ready
        if self.vectorizer.svd is None:
            self.vectorizer.load()

        svd_features = self.vectorizer.transform(texts)

        # 2. Load Model
        model = SiameseProjector(
            input_dim=self.config.METRIC_INPUT_DIM,
            hidden_dim=self.config.METRIC_HIDDEN_DIM,
            output_dim=self.config.METRIC_EMBEDDING_DIM,
            dropout=self.config.METRIC_DROPOUT,
        ).to(self.device)

        if load_cached_model and os.path.exists(self.config.METRIC_MODEL_PATH):
            model.load_state_dict(
                torch.load(self.config.METRIC_MODEL_PATH, map_location=self.device)
            )
        else:
            print("Warning: Loading untrained model or model file not found.")

        model.eval()

        # 3. Inference
        # Process in batches to avoid OOM
        batch_size = self.config.METRIC_BATCH_SIZE * 2
        embeddings = []

        # Convert to tensor dataset
        tensor_x = torch.tensor(svd_features, dtype=torch.float32)
        loader = DataLoader(
            tensor_x,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
        )

        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                out = model(batch_x)
                embeddings.append(out.cpu().numpy())

        return np.concatenate(embeddings, axis=0)


def run_stage2_training():
    """
    Helper to run the training pipeline.
    """
    stage2 = Stage2Metric(Config)
    stage2.train()
