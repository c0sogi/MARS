import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import numpy as np
import pandas as pd
import os
import gc

from library.config import Config
from library.dataset import ChemicalDataset
from library.utils import process_labels


class StoichiometryEncoder(nn.Module):
    """
    Encodes chemical images into a fixed-size embedding vector and predicts
    stoichiometry (atom counts) as a proxy task.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        embedding_dim=Config.EMBEDDING_DIM,
        num_atoms=Config.NUM_ATOMS,
    ):
        super().__init__()

        # Initialize backbone
        # num_classes=0 returns the pooled features (global average pooling usually)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        in_features = self.backbone.num_features

        # Bottleneck / Embedding Layer
        # Projects backbone features to the retrieval embedding space
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Regression Head
        # Predicts atom counts from the embedding
        self.head = nn.Linear(embedding_dim, num_atoms)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding_layer(features)
        atom_preds = self.head(embedding)
        return embedding, atom_preds


def train_encoder(debug=Config.DEBUG):
    """
    Trains the StoichiometryEncoder on the training set.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SAMPLE_SIZE]
        print(
            f"Debug mode: Training on {len(df_train)} samples, Validating on {len(df_val)} samples."
        )

    # 2. Prepare Datasets and Loaders
    train_dataset = ChemicalDataset(df_train, mode="train")
    val_dataset = ChemicalDataset(df_val, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model, Loss, Optimizer
    model = StoichiometryEncoder().to(Config.DEVICE)

    # SmoothL1Loss is robust for regression of counts
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            optimizer.zero_grad()
            _, preds = model(images)  # We only care about preds for training

            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(Config.DEVICE)
                targets = targets.to(Config.DEVICE)

                _, preds = model(images)
                loss = criterion(preds, targets)

                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_dataset)

        # Print metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # Cleanup
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()


def generate_submission(debug=Config.DEBUG):
    """
    Generates submission by:
    1. Creating an index of embeddings for the entire training set.
    2. Computing embeddings for the test set.
    3. Retrieving the nearest neighbor from the training set for each test image.
    4. Assigning the neighbor's InChI label to the test image.
    """
    print("Generating submission via Retrieval...")

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE]
        print(
            f"Debug mode: Indexing {len(df_train)} train samples, Predicting {len(df_test)} test samples."
        )

    # Load Model
    model = StoichiometryEncoder().to(Config.DEVICE)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print(
            "Warning: No trained model found. Using random weights (expect poor performance)."
        )

    model.eval()

    # ---------------------------------------------------------
    # 1. Build Training Index (Embeddings)
    # ---------------------------------------------------------
    # Use 'val' mode transforms to avoid augmentation noise during indexing
    train_dataset = ChemicalDataset(df_train, mode="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    train_embeddings = []

    print("Building training index...")
    with torch.no_grad():
        for images, _ in train_loader:  # Ignore targets
            images = images.to(Config.DEVICE)
            emb, _ = model(images)
            # Normalize for Cosine Similarity
            emb = F.normalize(emb, p=2, dim=1)
            train_embeddings.append(emb.cpu().numpy())

    train_embeddings = np.concatenate(train_embeddings, axis=0)  # (N_train, Dim)

    # Save embeddings and labels for potential re-use or debugging
    np.save(Config.TRAIN_EMBEDDINGS_PATH, train_embeddings)

    # Keep labels in memory for lookup
    train_labels = df_train["InChI"].values

    # Free up GPU memory
    del train_loader, train_dataset
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 2. Predict on Test Set via Retrieval
    # ---------------------------------------------------------
    test_dataset = ChemicalDataset(df_test, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    predictions = []
    image_ids = []

    # Convert train index to tensor on GPU for fast matrix multiplication
    # 1.5M * 256 floats ~ 1.5GB. A100 has 40GB. We are safe to put it all on GPU.
    index_tensor = torch.tensor(train_embeddings).to(Config.DEVICE)  # (N_train, Dim)

    print("Predicting test set...")
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(Config.DEVICE)

            # Get test embeddings
            test_emb, _ = model(images)
            test_emb = F.normalize(test_emb, p=2, dim=1)  # (Batch, Dim)

            # Compute Cosine Similarity: (Batch, Dim) @ (Dim, N_train) -> (Batch, N_train)
            scores = torch.matmul(test_emb, index_tensor.T)

            # Find nearest neighbor
            _, best_indices = torch.max(scores, dim=1)

            # Retrieve labels
            best_indices_cpu = best_indices.cpu().numpy()
            batch_preds = train_labels[best_indices_cpu]

            predictions.extend(batch_preds)
            image_ids.extend(ids)

    # ---------------------------------------------------------
    # 3. Save Submission
    # ---------------------------------------------------------
    submission_df = pd.DataFrame({"image_id": image_ids, "InChI": predictions})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
