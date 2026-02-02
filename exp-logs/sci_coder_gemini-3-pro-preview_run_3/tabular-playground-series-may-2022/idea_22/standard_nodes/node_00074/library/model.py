import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from library.config import Config


class AVPFE(nn.Module):
    """
    Anchor-Variant Parallel Funnel Ensemble (AV-PFE).
    Consists of 5 independent streams, each with its own embeddings and MLP.
    """

    def __init__(self, cat_cardinalities, n_cont_features):
        super(AVPFE, self).__init__()
        self.num_streams = len(Config.STREAM_CONFIGS)
        self.streams = nn.ModuleList()

        for stream_cfg in Config.STREAM_CONFIGS:
            # 1. Independent Embeddings
            # Create a list of embeddings, one for each categorical feature
            # This ensures each stream learns a unique representation
            embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=c, embedding_dim=Config.EMBED_DIM)
                    for c in cat_cardinalities
                ]
            )

            # Calculate input dimension for the MLP
            # Flattened embeddings + continuous features
            input_dim = (len(cat_cardinalities) * Config.EMBED_DIM) + n_cont_features

            # 2. Independent MLP (Funnel Architecture)
            layers = []
            curr_dim = input_dim

            # Hidden layers
            for hidden_dim in stream_cfg["layers"]:
                layers.append(nn.Linear(curr_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(stream_cfg["dropout"]))
                curr_dim = hidden_dim

            # Output projection to 1 unit (Logits)
            layers.append(nn.Linear(curr_dim, 1))

            # Store as a tuple-like structure within ModuleList
            # We store [embeddings_list, mlp_sequential]
            self.streams.append(nn.ModuleList([embeddings, nn.Sequential(*layers)]))

    def forward(self, x_cat, x_cont):
        """
        Forward pass for all 5 streams.
        x_cat: (Batch, N_Cat_Features) - LongTensor
        x_cont: (Batch, N_Cont_Features) - FloatTensor
        Returns: (Batch, 5) - Logits for each stream
        """
        outputs = []

        for i in range(self.num_streams):
            embeddings_list, mlp = self.streams[i]

            # Embed each categorical feature independently
            embedded_parts = []
            for j, emb_layer in enumerate(embeddings_list):
                # x_cat[:, j] is the column for the j-th categorical feature
                embedded_parts.append(emb_layer(x_cat[:, j]))

            # Concatenate all embeddings: (Batch, N_Cat * Embed_Dim)
            cat_embeds = torch.cat(embedded_parts, dim=1)

            # Concatenate with continuous features: (Batch, Total_Dim)
            full_input = torch.cat([cat_embeds, x_cont], dim=1)

            # Pass through MLP
            out = mlp(full_input)  # (Batch, 1)
            outputs.append(out)

        # Stack outputs along dim 1: (Batch, 5)
        return torch.cat(outputs, dim=1)


def train_one_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0.0

    for x_cat, x_cont, y in loader:
        x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x_cat, x_cont)  # (Batch, 5)

        # Loss is sum of BCE for each stream
        # y is (Batch, 1), outputs is (Batch, 5)
        loss = 0
        for i in range(5):
            loss += criterion(outputs[:, i : i + 1], y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_cat, x_cont, y in loader:
            x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)
            outputs = model(x_cat, x_cont)  # (Batch, 5)

            loss = 0
            for i in range(5):
                loss += criterion(outputs[:, i : i + 1], y)
            total_loss += loss.item()

            # Average probabilities for metric calculation
            # Sigmoid then Mean
            probs = torch.sigmoid(outputs).mean(dim=1)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate ROC AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return total_loss / len(loader), auc


def run_avpfe_pipeline(train_dataset, val_dataset, test_dataset):
    """
    Main execution function:
    1. Instantiates model based on dataset properties.
    2. Trains the model using OneCycleLR.
    3. Saves the best model based on Validation AUC.
    4. Generates predictions on the test set.
    5. Saves submission file.
    """
    print("Initializing AV-PFE Pipeline...")

    # 1. Determine Input Shapes
    # We need max index for each categorical column to set embedding sizes
    # Concatenate all x_cat to ensure we cover the full vocabulary
    full_cat = torch.cat(
        [train_dataset.x_cat, val_dataset.x_cat, test_dataset.x_cat], dim=0
    )
    # Max index + 1 is the cardinality
    cat_cardinalities = (full_cat.max(dim=0).values + 1).tolist()
    n_cont = train_dataset.x_cont.shape[1]

    print(f"Detected {len(cat_cardinalities)} categorical features.")
    print(f"Detected {n_cont} continuous features.")

    # 2. Setup Model and Training
    device = Config.DEVICE
    model = AVPFE(cat_cardinalities, n_cont).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # DataLoaders
    # Pin memory for faster transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 3. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        val_loss, val_auc = validate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f} | Val AUC: {val_auc:.10f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val AUC: {best_auc:.10f}")

    # 4. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    predictions = []

    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            outputs = model(x_cat, x_cont)
            # Average probabilities
            probs = torch.sigmoid(outputs).mean(dim=1)
            predictions.extend(probs.cpu().numpy())

    # 5. Save Submission
    # Load test IDs from metadata
    print("Generating submission file...")
    try:
        df_test = pd.read_csv(Config.TEST_METADATA_PATH, usecols=["id"])
        df_test["target"] = predictions
        df_test.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    except Exception as e:
        print(f"Error generating submission: {e}")
