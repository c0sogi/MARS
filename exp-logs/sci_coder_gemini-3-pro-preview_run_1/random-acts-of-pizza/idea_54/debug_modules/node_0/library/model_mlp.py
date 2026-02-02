import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from library.config import (
    MLP_HIDDEN_DIM,
    MLP_DROPOUT_EMB,
    MLP_DROPOUT_DENSE,
    MLP_BATCH_SIZE,
    MLP_LEARNING_RATE,
    MLP_WEIGHT_DECAY,
    MLP_EPOCHS,
    MLP_PATIENCE,
    DEVICE,
    WORKING_DIR,
    RANDOM_STATE,
)
from library.utils import set_seed, compute_auc

# Ensure deterministic behavior
set_seed(RANDOM_STATE)


class PizzaDataset(Dataset):
    """
    Custom Dataset to handle the multi-modal dictionary structure of the input data.
    """

    def __init__(self, features_dict, targets=None):
        """
        Args:
            features_dict (dict): Dictionary with keys 'semantic', 'reliability', 'community'.
                                  Values are numpy arrays.
            targets (np.ndarray, optional): Binary targets.
        """
        self.semantic = torch.FloatTensor(features_dict["semantic"])
        self.reliability = torch.FloatTensor(features_dict["reliability"])
        self.community = torch.FloatTensor(features_dict["community"])

        if targets is not None:
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None

    def __len__(self):
        return len(self.semantic)

    def __getitem__(self, idx):
        sample = {
            "semantic": self.semantic[idx],
            "reliability": self.reliability[idx],
            "community": self.community[idx],
        }

        if self.targets is not None:
            return sample, self.targets[idx]
        return sample, torch.tensor(-1.0)  # Dummy target for inference


class TopologyAwareMLP(nn.Module):
    """
    Topology-Aware Skip-Gated MLP.

    Structure:
    1. Reliability Branch -> Control Gate (Sigmoid)
    2. Semantic Branch -> Gated by Control Gate (Element-wise multiplication)
    3. Community Branch -> Skip Connection (Additive/Concatenated)
    4. Fusion -> MLP Head -> Output
    """

    def __init__(
        self,
        input_dim_semantic,
        input_dim_reliability,
        input_dim_community,
        hidden_dim=MLP_HIDDEN_DIM,
    ):
        super(TopologyAwareMLP, self).__init__()

        # 1. Gate Generator (Reliability -> Gate)
        # We project reliability to the same dimension as semantic to allow element-wise gating
        self.gate_projection = nn.Sequential(
            nn.Linear(input_dim_reliability, input_dim_semantic),
            nn.ReLU(),
            nn.Linear(input_dim_semantic, input_dim_semantic),
            nn.Sigmoid(),
        )

        # 2. Semantic Processing
        # Optional: A small projection for semantic features before gating if needed,
        # but here we apply gating directly to the raw semantic embeddings + consistency scalars
        self.semantic_dropout = nn.Dropout(MLP_DROPOUT_EMB)

        # 3. Fusion Layer
        # Input: Gated Semantic (dim_sem) + Reliability (dim_rel) + Community (dim_com)
        fusion_input_dim = (
            input_dim_semantic + input_dim_reliability + input_dim_community
        )

        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch Norm helps with convergence
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT_DENSE),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT_DENSE),
            nn.Linear(hidden_dim // 2, 1),  # Logits
        )

    def forward(self, semantic, reliability, community):
        # Apply dropout to semantic embeddings
        sem_emb = self.semantic_dropout(semantic)

        # Generate Gate from Reliability
        gate = self.gate_projection(reliability)

        # Apply Gate (Topology-Aware Modulation)
        # "Credibility modulates Relevance"
        gated_semantic = sem_emb * gate

        # Concatenate: Gated Semantic + Raw Reliability + Community Skip Connection
        # Community acts as a direct additive bias/signal
        combined = torch.cat([gated_semantic, reliability, community], dim=1)

        # MLP Head
        logits = self.head(combined)
        return logits


def train_mlp_model(mlp_features, targets):
    """
    Trains the Topology-Aware MLP (Stream B).

    Args:
        mlp_features (dict): Dictionary containing 'train', 'val', 'test' sub-dictionaries.
                             Each sub-dict has keys 'semantic', 'reliability', 'community'.
        targets (dict): Dictionary containing 'train' and 'val' target arrays.

    Returns:
        tuple: (val_preds, test_preds, model)
    """
    print("Initializing Topology-Aware MLP (Stream B)...")

    # 1. Prepare DataLoaders
    train_dataset = PizzaDataset(mlp_features["train"], targets["train"])
    val_dataset = PizzaDataset(mlp_features["val"], targets["val"])
    test_dataset = PizzaDataset(mlp_features["test"], None)

    train_loader = DataLoader(
        train_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=MLP_BATCH_SIZE * 2, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=MLP_BATCH_SIZE * 2, shuffle=False, num_workers=0
    )

    # 2. Determine Input Dimensions
    # Inspect one batch to get shapes
    sample_sem = mlp_features["train"]["semantic"]
    sample_rel = mlp_features["train"]["reliability"]
    sample_com = mlp_features["train"]["community"]

    input_dim_semantic = sample_sem.shape[1]
    input_dim_reliability = sample_rel.shape[1]
    input_dim_community = sample_com.shape[1]

    print(
        f"MLP Input Dims - Semantic: {input_dim_semantic}, Reliability: {input_dim_reliability}, Community: {input_dim_community}"
    )

    # 3. Initialize Model
    model = TopologyAwareMLP(
        input_dim_semantic, input_dim_reliability, input_dim_community
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY
    )

    # 4. Training Loop with Early Stopping
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training on device: {DEVICE}")

    for epoch in range(MLP_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_data, batch_targets in train_loader:
            sem = batch_data["semantic"].to(DEVICE)
            rel = batch_data["reliability"].to(DEVICE)
            com = batch_data["community"].to(DEVICE)
            y = batch_targets.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(sem, rel, com)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch_data, batch_targets in val_loader:
                sem = batch_data["semantic"].to(DEVICE)
                rel = batch_data["reliability"].to(DEVICE)
                com = batch_data["community"].to(DEVICE)

                logits = model(sem, rel, com)
                probs = torch.sigmoid(logits)

                val_preds_list.extend(probs.cpu().numpy())
                val_targets_list.extend(batch_targets.numpy())

        val_preds_arr = np.array(val_preds_list).flatten()
        val_targets_arr = np.array(val_targets_list).flatten()

        val_auc = compute_auc(val_targets_arr, val_preds_arr)

        # print(f"Epoch {epoch+1}/{MLP_EPOCHS} | Loss: {train_loss:.4f} | Val AUC: {val_auc}")

        # Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= MLP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 5. Finalize
    print(f"Best MLP Validation AUC: {best_val_auc}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model
    os.makedirs(WORKING_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(WORKING_DIR, "best_mlp.pth"))

    # 6. Generate Predictions
    model.eval()

    # Re-generate val preds with best model
    final_val_preds = []
    with torch.no_grad():
        for batch_data, _ in val_loader:
            sem = batch_data["semantic"].to(DEVICE)
            rel = batch_data["reliability"].to(DEVICE)
            com = batch_data["community"].to(DEVICE)
            probs = torch.sigmoid(model(sem, rel, com))
            final_val_preds.extend(probs.cpu().numpy())

    # Generate test preds
    test_preds = []
    with torch.no_grad():
        for batch_data, _ in test_loader:
            sem = batch_data["semantic"].to(DEVICE)
            rel = batch_data["reliability"].to(DEVICE)
            com = batch_data["community"].to(DEVICE)
            probs = torch.sigmoid(model(sem, rel, com))
            test_preds.extend(probs.cpu().numpy())

    final_val_preds = np.array(final_val_preds).flatten()
    final_test_preds = np.array(test_preds).flatten()

    # Save predictions
    np.savez(
        os.path.join(WORKING_DIR, "mlp_preds.npz"),
        val=final_val_preds,
        test=final_test_preds,
    )

    return final_val_preds, final_test_preds, model
