import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import (
    MLP_PARAMS,
    WORKING_DIR,
    TARGET_COL,
    DEVICE,
    RANDOM_SEED,
    SBERT_EMBEDDING_DIM,
    MAX_HISTORY_LEN,
)
from library.utils import ensure_directory, seed_everything, calculate_auc
from library.feature_engineering import MetadataExtractor, SBERTHandler


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Gated Fusion MLP.
    Serves Request Embeddings, History Sequences, History Masks, Metadata, and Labels.
    """

    def __init__(
        self,
        request_emb: np.ndarray,
        history_emb: np.ndarray,
        meta_features: np.ndarray,
        labels: np.ndarray = None,
    ):
        self.request_emb = torch.FloatTensor(request_emb)
        self.history_emb = torch.FloatTensor(history_emb)
        self.meta_features = torch.FloatTensor(meta_features)

        # Create mask: 1 for valid tokens, 0 for padding (where vector is all zeros)
        # Check L1 norm of embedding vectors to detect padding
        history_norm = np.abs(history_emb).sum(axis=-1)
        self.history_mask = torch.FloatTensor((history_norm > 1e-6).astype(np.float32))

        if labels is not None:
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.request_emb)

    def __getitem__(self, idx):
        item = {
            "request_emb": self.request_emb[idx],
            "history_emb": self.history_emb[idx],
            "history_mask": self.history_mask[idx],
            "meta_features": self.meta_features[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class MaskedAttention(nn.Module):
    """
    Dot-Product Attention with Explicit Additive Masking.
    Ensures padding tokens receive -inf score before Softmax.
    """

    def __init__(self, embed_dim):
        super(MaskedAttention, self).__init__()
        self.scale = embed_dim**-0.5

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (Batch, Embed_Dim)
            key: (Batch, Seq_Len, Embed_Dim)
            value: (Batch, Seq_Len, Embed_Dim)
            mask: (Batch, Seq_Len) - 1 for valid, 0 for pad
        """
        # Expand query to (Batch, 1, Embed_Dim)
        query = query.unsqueeze(1)

        # Calculate scores: (Batch, 1, Seq_Len)
        # Q * K^T
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Expand mask to (Batch, 1, Seq_Len)
            mask = mask.unsqueeze(1)

            # Additive Masking: Fill 0 positions with -1e9
            # We use masked_fill on the boolean inverse of the mask
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Handle case where entire sequence is masked (e.g. no history)
        # If mask is all 0s, softmax output might be uniform or NaN.
        # We multiply by mask again to ensure output is 0 for purely padded sequences.
        if mask is not None:
            attn_weights = attn_weights * mask
            # Normalize again if needed, but usually 0 vector is fine for empty history

        # Weighted Sum: (Batch, 1, Embed_Dim)
        context = torch.matmul(attn_weights, value)

        # Squeeze back to (Batch, Embed_Dim)
        return context.squeeze(1)


class GatedFusionNet(nn.Module):
    """
    Neural Network with 3 Branches:
    1. Request Semantics (Raw SBERT)
    2. History Semantics (Masked Attention)
    3. Metadata (Dense -> Gate)

    Fusion: (Request || History) * Gate(Metadata)
    """

    def __init__(
        self, meta_dim, embed_dim=SBERT_EMBEDDING_DIM, hidden_dim=128, dropout=0.3
    ):
        super(GatedFusionNet, self).__init__()

        # Branch 2: Attention Mechanism
        self.attention = MaskedAttention(embed_dim)

        # Branch 3: Metadata Gate
        # Projects metadata to size of concatenated semantic vector (2 * embed_dim)
        self.meta_gate = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * embed_dim),
            nn.Sigmoid(),
        )

        # Classifier Head
        # Input: Concatenated Semantic Vector (2 * embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Logits output
        )

    def forward(self, request_emb, history_emb, history_mask, meta_features):
        # 1. Request Branch (Pass-through raw embedding)
        # request_emb: (Batch, Embed_Dim)

        # 2. History Branch (Attention)
        # history_context: (Batch, Embed_Dim)
        history_context = self.attention(
            query=request_emb, key=history_emb, value=history_emb, mask=history_mask
        )

        # 3. Concatenate Semantics
        # semantic_vector: (Batch, 2 * Embed_Dim)
        semantic_vector = torch.cat([request_emb, history_context], dim=1)

        # 4. Metadata Gating
        # gate: (Batch, 2 * Embed_Dim)
        gate = self.meta_gate(meta_features)

        # Apply Gate
        gated_semantics = semantic_vector * gate

        # 5. Classification
        logits = self.classifier(gated_semantics)
        return logits


class MLPModel:
    """
    Stream B: Masked-Attention Gated MLP.
    Wraps feature engineering, preprocessing, and PyTorch training loop.
    """

    def __init__(self):
        self.metadata_extractor = MetadataExtractor()
        self.sbert_handler = SBERTHandler()
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device(DEVICE)

        # Hyperparameters
        self.params = MLP_PARAMS

    def _process_features(
        self,
        df: pd.DataFrame,
        split_name: str,
        is_training: bool = False,
        load_cached_data: bool = True,
    ):
        """
        Extracts SBERT and Metadata features.
        Applies Arcsinh + StandardScaler to metadata.
        Caches the processed numpy arrays.
        """
        cache_file = f"mlp_data_{split_name}.npz"
        cache_path = os.path.join(WORKING_DIR, cache_file)

        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached MLP data from {cache_path}")
            data = np.load(cache_path)
            # If training, we must ensure scaler is fitted.
            # In a real scenario, we'd save the scaler. Here, we assume if cache exists,
            # we might need to re-fit scaler on loaded train data if is_training=True.
            # For simplicity in this script, if is_training=True, we re-fit scaler on the loaded data.
            meta = data["meta"]
            if is_training:
                self.scaler.fit(meta)
            return data["req"], data["hist"], meta, data["y"] if "y" in data else None

        print(f"Processing MLP features for {split_name}...")

        # 1. SBERT Features
        req_emb = self.sbert_handler.encode_requests(df, split_name, load_cached_data)
        hist_emb = self.sbert_handler.encode_history(df, split_name, load_cached_data)

        # 2. Metadata Features
        meta_df = self.metadata_extractor.process(df, split_name, load_cached_data)
        meta_values = meta_df.values.astype(np.float32)

        # 3. Preprocessing (Arcsinh)
        # Apply arcsinh to handle skewed distributions (counts, karma)
        meta_values = np.arcsinh(meta_values)

        # 4. Scaling
        # We fit on training data, transform on others.
        # Note: This logic requires that we process 'train' first in the pipeline.
        if is_training:
            self.scaler.fit(meta_values)
            meta_scaled = self.scaler.transform(meta_values)
        else:
            # Check if scaler is fitted
            try:
                meta_scaled = self.scaler.transform(meta_values)
            except:
                # Fallback if scaler not fitted (e.g. separate run), though ideally shouldn't happen
                print(
                    "Warning: Scaler not fitted. Fitting on current batch (Suboptimal)."
                )
                self.scaler.fit(meta_values)
                meta_scaled = self.scaler.transform(meta_values)

        # 5. Target
        y = None
        if TARGET_COL in df.columns:
            y = df[TARGET_COL].values.astype(np.float32)

        # Cache
        ensure_directory(cache_path)
        save_dict = {"req": req_emb, "hist": hist_emb, "meta": meta_scaled}
        if y is not None:
            save_dict["y"] = y

        np.savez(cache_path, **save_dict)
        print(f"Saved MLP data to {cache_path}")

        return req_emb, hist_emb, meta_scaled, y

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        load_cached_data: bool = True,
    ):
        """
        Trains the GatedFusionNet.
        """
        seed_everything(RANDOM_SEED)

        # 1. Prepare Data
        print("Preparing MLP Training Data...")
        train_req, train_hist, train_meta, train_y = self._process_features(
            train_df, "train", is_training=True, load_cached_data=load_cached_data
        )

        print("Preparing MLP Validation Data...")
        val_req, val_hist, val_meta, val_y = self._process_features(
            val_df, "val", is_training=False, load_cached_data=load_cached_data
        )

        # Create Datasets & Loaders
        train_dataset = PizzaDataset(train_req, train_hist, train_meta, train_y)
        val_dataset = PizzaDataset(val_req, val_hist, val_meta, val_y)

        train_loader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.params["batch_size"], shuffle=False
        )

        # 2. Initialize Model
        meta_dim = train_meta.shape[1]
        self.model = GatedFusionNet(
            meta_dim=meta_dim,
            embed_dim=self.params["embedding_dim"],
            hidden_dim=self.params["hidden_dim"],
            dropout=self.params["dropout"],
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )

        # 3. Training Loop
        print(f"Starting MLP Training on {self.device}...")
        best_auc = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.params["epochs"]):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                req = batch["request_emb"].to(self.device)
                hist = batch["history_emb"].to(self.device)
                mask = batch["history_mask"].to(self.device)
                meta = batch["meta_features"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits = self.model(req, hist, mask, meta)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * req.size(0)

            train_loss /= len(train_dataset)

            # Validation
            self.model.eval()
            val_preds = []
            val_targets = []
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    req = batch["request_emb"].to(self.device)
                    hist = batch["history_emb"].to(self.device)
                    mask = batch["history_mask"].to(self.device)
                    meta = batch["meta_features"].to(self.device)
                    labels = batch["label"].to(self.device).unsqueeze(1)

                    logits = self.model(req, hist, mask, meta)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * req.size(0)

                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(probs)
                    val_targets.extend(labels.cpu().numpy())

            val_loss /= len(val_dataset)
            val_auc = calculate_auc(val_targets, val_preds)

            print(
                f"Epoch {epoch+1}/{self.params['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc}"
            )

            # Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.params["patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Best Validation AUC: {best_auc}")

        return best_auc

    def predict_proba(
        self, test_df: pd.DataFrame, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Generates predictions for the test set.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        print("Preparing MLP Test Data...")
        test_req, test_hist, test_meta, _ = self._process_features(
            test_df, "test", is_training=False, load_cached_data=load_cached_data
        )

        dataset = PizzaDataset(test_req, test_hist, test_meta, None)
        loader = DataLoader(
            dataset, batch_size=self.params["batch_size"], shuffle=False
        )

        self.model.eval()
        all_preds = []

        print("Generating MLP Predictions...")
        with torch.no_grad():
            for batch in loader:
                req = batch["request_emb"].to(self.device)
                hist = batch["history_emb"].to(self.device)
                mask = batch["history_mask"].to(self.device)
                meta = batch["meta_features"].to(self.device)

                logits = self.model(req, hist, mask, meta)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)

        return np.array(all_preds).flatten()
