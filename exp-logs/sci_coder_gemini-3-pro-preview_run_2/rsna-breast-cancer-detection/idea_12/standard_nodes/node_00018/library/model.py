import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from library import config, utils


class MRHNModel(nn.Module):
    """
    Modality-Regularized Hybrid Network (MR-HN).
    Combines EfficientNetV2-S for images and an MLP for tabular data.
    Uses Modality Dropout to prevent overfitting to tabular priors.
    """

    def __init__(self, vocab_sizes):
        super().__init__()

        # 1. Visual Backbone
        # efficientnet_v2_s: Pretrained, no classifier head
        self.backbone = timm.create_model(
            config.MODEL_NAME,
            pretrained=config.PRETRAINED,
            num_classes=0,
            drop_path_rate=config.DROP_PATH_RATE,
            in_chans=config.NUM_CHANNELS,
        )
        self.img_dim = self.backbone.num_features

        # 2. Tabular Branch
        self.cat_cols = config.CATEGORICAL_COLS
        self.num_cols = config.NUMERICAL_COLS

        # Categorical Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_sizes[col], config.TABULAR_EMBED_DIM)
                for col in self.cat_cols
            ]
        )

        # Calculate input dimension for Tabular MLP
        # Sum of all embedding dims + number of numerical features
        self.tabular_input_dim = (len(self.cat_cols) * config.TABULAR_EMBED_DIM) + len(
            self.num_cols
        )

        # Tabular MLP
        self.tabular_mlp = nn.Sequential(
            nn.Linear(self.tabular_input_dim, config.TABULAR_HIDDEN_DIM),
            nn.BatchNorm1d(config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.TABULAR_HIDDEN_DIM, config.TABULAR_HIDDEN_DIM),
            nn.BatchNorm1d(config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
        )

        # 3. Fusion & Head
        # Concatenation of Image features and Tabular features
        self.head = nn.Sequential(
            nn.Dropout(config.DROP_RATE),
            nn.Linear(self.img_dim + config.TABULAR_HIDDEN_DIM, 1),
        )

        self.modality_dropout_prob = config.MODALITY_DROPOUT_PROB

    def forward(self, images, tabular_data):
        """
        Args:
            images: Tensor (B, C, H, W)
            tabular_data: Tuple (cat_feats, num_feats)
                cat_feats: Tensor (B, Num_Cats)
                num_feats: Tensor (B, Num_Nums)
        """
        cat_feats, num_feats = tabular_data

        # --- Visual Forward ---
        img_feats = self.backbone(images)  # (B, img_dim)

        # --- Tabular Forward ---
        # Look up embeddings
        embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            embeds.append(emb_layer(cat_feats[:, i]))

        # Concat embeddings (B, Total_Embed_Dim)
        cat_embeds = torch.cat(embeds, dim=1)

        # Concat with numerical features
        tab_input = torch.cat([cat_embeds, num_feats], dim=1)

        # MLP
        tab_feats = self.tabular_mlp(tab_input)  # (B, TABULAR_HIDDEN_DIM)

        # --- Modality Dropout ---
        # Randomly zero out tabular features during training
        if self.training:
            # Probability of keeping the modality
            keep_prob = 1.0 - self.modality_dropout_prob

            # Generate Bernoulli mask (B, 1)
            mask = torch.bernoulli(
                torch.full((tab_feats.size(0), 1), keep_prob, device=tab_feats.device)
            )

            # Apply mask and scale to preserve expected magnitude
            tab_feats = tab_feats * mask * (1.0 / keep_prob)

        # --- Fusion ---
        combined = torch.cat([img_feats, tab_feats], dim=1)

        # --- Prediction ---
        logits = self.head(combined)

        return logits


def train_model(train_loader, val_loader, feature_meta):
    """
    Trains the MRHNModel.
    """
    device = config.DEVICE
    vocab_sizes = feature_meta["vocab_sizes"]

    # Initialize Model
    model = MRHNModel(vocab_sizes).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    # Loss Function
    # High positive weight for class imbalance
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler("cuda")

    best_pf1 = 0.0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0

        # Training Loop
        for images, (cat_feats, num_feats), labels in train_loader:
            images = images.to(device)
            cat_feats = cat_feats.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()

            # Forward pass with AMP
            with torch.amp.autocast("cuda"):
                logits = model(images, (cat_feats, num_feats))

                # Compute Loss
                if config.USE_FP32_LOSS:
                    # Exit AMP for loss calculation to prevent NaN with high pos_weight
                    with torch.amp.autocast("cuda", enabled=False):
                        loss = criterion(logits.float(), labels.float())
                else:
                    loss = criterion(logits, labels)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, (cat_feats, num_feats), labels in val_loader:
                images = images.to(device)
                cat_feats = cat_feats.to(device)
                num_feats = num_feats.to(device)

                with torch.amp.autocast("cuda"):
                    logits = model(images, (cat_feats, num_feats))
                    probs = torch.sigmoid(logits)

                val_preds.extend(probs.cpu().numpy().flatten())
                val_labels.extend(labels.numpy().flatten())

        # Metric Calculation
        pf1 = utils.pf1_score(val_labels, val_preds)

        print(f"Epoch {epoch+1} | Loss: {avg_train_loss} | Val pF1: {pf1}")

        # Save Best Model
        if pf1 > best_pf1:
            best_pf1 = pf1
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved Best Model (pF1: {pf1})")

    return best_model_path


def predict_and_submit(model_path, test_loader, feature_meta):
    """
    Generates predictions and creates the submission file.
    """
    device = config.DEVICE
    vocab_sizes = feature_meta["vocab_sizes"]

    # Load Model
    model = MRHNModel(vocab_sizes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []

    # Get prediction IDs from dataset (order is preserved as shuffle=False)
    prediction_ids = test_loader.dataset.prediction_ids

    print("Generating predictions...")
    with torch.no_grad():
        for images, (cat_feats, num_feats), _ in test_loader:
            images = images.to(device)
            cat_feats = cat_feats.to(device)
            num_feats = num_feats.to(device)

            with torch.amp.autocast("cuda"):
                logits = model(images, (cat_feats, num_feats))
                probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())

    # Create DataFrame
    df_pred = pd.DataFrame({"prediction_id": prediction_ids, "cancer": all_preds})

    # Aggregate by prediction_id (Max Pooling)
    # This handles multiple views per breast
    submission = df_pred.groupby("prediction_id")["cancer"].max().reset_index()

    # Save Submission
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
