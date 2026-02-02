import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import timm
from sklearn.metrics import roc_auc_score
from library.utils import set_seed, get_device
from library.data_loader import get_dataloaders


class SiameseNetwork(nn.Module):
    """
    Siamese 2.5D Convolutional Neural Network with Spatial Feature Fusion.

    Architecture:
    - Backbone: EfficientNet-B0 (Shared Weights)
    - Input: Two streams (Even slices, Odd slices), each (B, 64, 224, 224)
    - Fusion: Spatial concatenation of feature maps followed by 1x1 Conv compression
    - Head: Global Average Pooling + Linear
    """

    def __init__(
        self, model_name="efficientnet_b0", pretrained=True, drop_path_rate=0.2
    ):
        super(SiameseNetwork, self).__init__()

        # Shared backbone
        # in_chans=64 to accommodate the stacked modality slices (16 slices * 4 modalities)
        # drop_path_rate=0.2 for regularization (Stochastic Depth)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=64,
            drop_path_rate=drop_path_rate,
            features_only=True,
            out_indices=(4,),  # Extract the last feature map (e.g., 7x7x1280)
        )

        # Get feature dimension (e.g., 1280 for efficientnet_b0)
        feature_dim = self.backbone.feature_info[-1]["num_chs"]

        # Spatial Fusion Head
        # Concatenates features from both streams (2 * feature_dim) and compresses back to feature_dim
        # This preserves spatial information before global pooling
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, 1)

    def forward_one(self, x):
        """Passes one stream through the backbone."""
        features = self.backbone(x)
        return features[-1]  # Return the last feature map

    def forward(self, x_even, x_odd):
        # Shared weights: process both streams
        f_even = self.forward_one(x_even)
        f_odd = self.forward_one(x_odd)

        # Spatial Feature Fusion
        # Concatenate along channel dimension
        concat = torch.cat([f_even, f_odd], dim=1)

        # Compress channels and learn local spatial correlations
        fused = self.fusion_conv(concat)

        # Classification
        pooled = self.global_pool(fused).flatten(1)
        logits = self.fc(pooled)

        return logits


def train_and_predict(
    train_meta_path="./metadata/train.parquet",
    val_meta_path="./metadata/val.parquet",
    test_meta_path="./metadata/test.parquet",
    submission_path="./submission/submission.csv",
    cache_dir="./working/idea_40/",
    epochs=15,
    batch_size=16,
    lr=1e-4,
    seed=42,
):
    """
    Main execution function to train the SiameseNetwork and generate submission.
    """
    # Set reproducibility
    set_seed(seed)
    device = get_device()

    # 1. Data Loading
    # Uses the library data loader which handles caching and processing
    train_loader, val_loader, test_loader = get_dataloaders(
        train_meta_path=train_meta_path,
        val_meta_path=val_meta_path,
        test_meta_path=test_meta_path,
        batch_size=batch_size,
        num_workers=4,
        load_cached_data=True,
        cache_dir=cache_dir,
    )

    # 2. Model Initialization
    model = SiameseNetwork(
        model_name="efficientnet_b0", pretrained=True, drop_path_rate=0.2
    )
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training Loop with Early Stopping
    best_auc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(cache_dir, "best_model_siamese.pth")

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        if train_loader:
            for xe, xo, labels in train_loader:
                xe, xo, labels = xe.to(device), xo.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(xe, xo).squeeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * xe.size(0)
                train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                train_targets.extend(labels.cpu().numpy())

            train_loss /= len(train_loader.dataset)
            train_auc = roc_auc_score(train_targets, train_preds)
        else:
            train_loss = 0.0
            train_auc = 0.0

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        if val_loader:
            with torch.no_grad():
                for xe, xo, labels in val_loader:
                    xe, xo, labels = xe.to(device), xo.to(device), labels.to(device)
                    outputs = model(xe, xo).squeeze(1)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * xe.size(0)
                    val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_auc = roc_auc_score(val_targets, val_preds)
        else:
            val_loss = 0.0
            val_auc = 0.0

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.10f} | Train AUC: {train_auc:.10f} | "
            f"Val Loss: {val_loss:.10f} | Val AUC: {val_auc:.10f}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 4. Inference
    print("Starting inference on test set...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    predictions = []

    if test_loader:
        with torch.no_grad():
            for batch in test_loader:
                # Unpack based on dataset return (SiameseDataset returns 2 items if y is None)
                if len(batch) == 2:
                    xe, xo = batch
                else:
                    xe, xo, _ = batch

                xe, xo = xe.to(device), xo.to(device)
                outputs = model(xe, xo).squeeze(1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                predictions.extend(probs)

        # Retrieve IDs from cached file to ensure alignment
        ids_path = os.path.join(cache_dir, "ids_test.npy")
        if os.path.exists(ids_path):
            test_ids = np.load(ids_path, allow_pickle=True)
        else:
            # Fallback
            df_test = pd.read_parquet(test_meta_path)
            test_ids = df_test["BraTS21ID"].values

        # Generate Submission
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": predictions})
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
