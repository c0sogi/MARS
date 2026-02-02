import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import MLP_PARAMS, WORKING_DIR, SEED

# Set seeds for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


class PizzaDataset(Dataset):
    """
    Custom Dataset to handle dictionary-based inputs for the MLP.
    Wraps pre-computed embeddings and metadata.
    """

    def __init__(self, features_dict, split_name):
        self.metadata = torch.tensor(
            features_dict[f"{split_name}_metadata"], dtype=torch.float32
        )
        self.title_emb = torch.tensor(
            features_dict[f"{split_name}_title_emb"], dtype=torch.float32
        )
        self.body_emb = torch.tensor(
            features_dict[f"{split_name}_body_emb"], dtype=torch.float32
        )
        self.hist_centroid = torch.tensor(
            features_dict[f"{split_name}_hist_centroid"], dtype=torch.float32
        )
        self.hist_seq = torch.tensor(
            features_dict[f"{split_name}_hist_seq"], dtype=torch.float32
        )
        self.targets = torch.tensor(
            features_dict[f"{split_name}_target"], dtype=torch.float32
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "metadata": self.metadata[idx],
            "title_emb": self.title_emb[idx],
            "body_emb": self.body_emb[idx],
            "hist_centroid": self.hist_centroid[idx],
            "hist_seq": self.hist_seq[idx],
            "target": self.targets[idx],
        }


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Predicts affine parameters (gamma, beta) from conditioning input z
    to modulate input x: out = (1 + gamma) * x + beta.
    """

    def __init__(self, input_dim, conditioning_dim, film_dim=128):
        super(FiLMLayer, self).__init__()
        # Generator network to predict modulation parameters
        self.generator = nn.Sequential(
            nn.Linear(conditioning_dim, film_dim),
            nn.ReLU(),
            nn.Linear(film_dim, input_dim * 2),
        )

        # Initialize the last layer to zero so the initial state is identity
        # gamma starts at 0 (scale factor 1), beta starts at 0
        nn.init.constant_(self.generator[-1].weight, 0.0)
        nn.init.constant_(self.generator[-1].bias, 0.0)

    def forward(self, x, z):
        # x: [batch, input_dim]
        # z: [batch, conditioning_dim]
        params = self.generator(z)
        gamma, beta = torch.split(params, x.shape[1], dim=1)

        return (1.0 + gamma) * x + beta


class DualQueryAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    Queries the user history sequence using a specific query vector (Title or Body).
    """

    def __init__(self, embed_dim):
        super(DualQueryAttention, self).__init__()
        self.scale = embed_dim**-0.5

    def forward(self, query, key_values):
        # query: [batch, embed_dim]
        # key_values: [batch, seq_len, embed_dim]

        q = query.unsqueeze(1)  # [batch, 1, embed_dim]
        k = key_values
        v = key_values

        # Compute attention scores: [batch, 1, seq_len]
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

        # Masking: Identify padding (where embedding vector is all zeros)
        # [batch, seq_len]
        is_padding = key_values.abs().sum(dim=2) == 0
        mask = is_padding.unsqueeze(1)  # [batch, 1, seq_len]

        # Apply mask (set padding scores to -inf)
        scores = scores.masked_fill(mask, -1e9)

        attn_weights = torch.softmax(scores, dim=2)

        # Compute context: [batch, 1, embed_dim]
        context = torch.bmm(attn_weights, v)

        return context.squeeze(1)


class FiLMConditionedMLP(nn.Module):
    """
    Main Neural Network Architecture.
    Combines Dual-Query Attention on history with FiLM modulation based on metadata.
    """

    def __init__(
        self,
        metadata_dim,
        embed_dim=384,
        hidden_dim=256,
        film_dim=128,
        dropout_emb=0.5,
        dropout_dense=0.2,
    ):
        super(FiLMConditionedMLP, self).__init__()

        self.attention = DualQueryAttention(embed_dim)
        self.dropout_emb = nn.Dropout(dropout_emb)

        # Input construction:
        # Title(384) + Body(384) + Ctx_Title(384) + Ctx_Body(384) + Centroid(384) + Cons_Title(1) + Cons_Body(1)
        self.concat_dim = (embed_dim * 5) + 2

        # FiLM Layer for modulation
        self.film = FiLMLayer(self.concat_dim, metadata_dim, film_dim=film_dim)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, title, body, hist_seq, hist_centroid, metadata):
        # 1. Dual-Query Attention
        ctx_title = self.attention(title, hist_seq)
        ctx_body = self.attention(body, hist_seq)

        # 2. Consistency Scalars (Dot Product)
        # Compute alignment between current request and history centroid
        cons_title = torch.sum(title * hist_centroid, dim=1, keepdim=True)
        cons_body = torch.sum(body * hist_centroid, dim=1, keepdim=True)

        # 3. Concatenation
        x = torch.cat(
            [title, body, ctx_title, ctx_body, hist_centroid, cons_title, cons_body],
            dim=1,
        )

        x = self.dropout_emb(x)

        # 4. FiLM Modulation
        # Modulate semantic features x using metadata z
        x = self.film(x, metadata)

        # 5. Classification
        logits = self.classifier(x)
        return logits


class MLPTrainer:
    """
    Trainer class to handle training, validation, and inference for the MLP.
    """

    def __init__(self):
        self.device = torch.device(
            MLP_PARAMS["device"] if torch.cuda.is_available() else "cpu"
        )
        print(f"MLP Model initialized on {self.device}")
        self.model = None
        self.model_path = os.path.join(WORKING_DIR, "best_mlp.pth")

    def train(self, features_dict):
        # Prepare Datasets and Loaders
        train_dataset = PizzaDataset(features_dict, "train")
        val_dataset = PizzaDataset(features_dict, "val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=MLP_PARAMS["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=MLP_PARAMS["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        # Infer metadata dimension from data
        sample_batch = next(iter(train_loader))
        metadata_dim = sample_batch["metadata"].shape[1]

        # Initialize Model
        self.model = FiLMConditionedMLP(
            metadata_dim=metadata_dim,
            embed_dim=384,
            hidden_dim=MLP_PARAMS["hidden_dim"],
            film_dim=MLP_PARAMS["film_dim"],
            dropout_emb=MLP_PARAMS["dropout_emb"],
            dropout_dense=MLP_PARAMS["dropout_dense"],
        ).to(self.device)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=MLP_PARAMS["learning_rate"],
            weight_decay=MLP_PARAMS["weight_decay"],
        )
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        patience_counter = 0

        print(f"Starting MLP training for {MLP_PARAMS['epochs']} epochs...")

        for epoch in range(MLP_PARAMS["epochs"]):
            self.model.train()
            train_loss = 0.0
            all_preds = []
            all_targets = []

            for batch in train_loader:
                # Move data to device
                metadata = batch["metadata"].to(self.device)
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist_seq = batch["hist_seq"].to(self.device)
                hist_centroid = batch["hist_centroid"].to(self.device)
                targets = batch["target"].to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits = self.model(title, body, hist_seq, hist_centroid, metadata)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * targets.size(0)
                all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())

            train_loss /= len(train_dataset)
            train_auc = roc_auc_score(all_targets, all_preds)

            # Validation
            val_loss, val_auc = self.evaluate(val_loader, criterion)

            print(
                f"Epoch {epoch+1}/{MLP_PARAMS['epochs']} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
            else:
                patience_counter += 1
                if patience_counter >= MLP_PARAMS["patience"]:
                    print("Early stopping triggered.")
                    break

        print(f"Best MLP Validation AUC: {best_auc}")
        # Reload best model weights
        self.load()

    def evaluate(self, dataloader, criterion=None):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for batch in dataloader:
                metadata = batch["metadata"].to(self.device)
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist_seq = batch["hist_seq"].to(self.device)
                hist_centroid = batch["hist_centroid"].to(self.device)
                targets = batch["target"].to(self.device).unsqueeze(1)

                logits = self.model(title, body, hist_seq, hist_centroid, metadata)
                loss = criterion(logits, targets)

                total_loss += loss.item() * targets.size(0)
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        auc = roc_auc_score(all_targets, all_preds)
        return avg_loss, auc

    def predict_proba(self, features_dict, split_name="test"):
        dataset = PizzaDataset(features_dict, split_name)
        dataloader = DataLoader(
            dataset, batch_size=MLP_PARAMS["batch_size"], shuffle=False, num_workers=0
        )

        # Ensure model is initialized if predicting without training in this session
        if self.model is None:
            sample_batch = next(iter(dataloader))
            metadata_dim = sample_batch["metadata"].shape[1]
            self.model = FiLMConditionedMLP(
                metadata_dim=metadata_dim,
                embed_dim=384,
                hidden_dim=MLP_PARAMS["hidden_dim"],
                film_dim=MLP_PARAMS["film_dim"],
                dropout_emb=MLP_PARAMS["dropout_emb"],
                dropout_dense=MLP_PARAMS["dropout_dense"],
            ).to(self.device)
            self.load()

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                metadata = batch["metadata"].to(self.device)
                title = batch["title_emb"].to(self.device)
                body = batch["body_emb"].to(self.device)
                hist_seq = batch["hist_seq"].to(self.device)
                hist_centroid = batch["hist_centroid"].to(self.device)

                logits = self.model(title, body, hist_seq, hist_centroid, metadata)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)

        return np.array(all_preds).flatten()

    def load(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            print(f"MLP model loaded from {self.model_path}")
            return True
        else:
            print(f"No saved MLP model found at {self.model_path}")
            return False
