import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from library import config, data_loader, feature_engineering, text_processing


# =============================================================================
# DATASET
# =============================================================================
class PizzaDataset(Dataset):
    def __init__(self, request_emb, history_emb, metadata, labels=None):
        """
        Args:
            request_emb: (N, 384)
            history_emb: (N, SeqLen, 384)
            metadata: (N, MetaDim)
            labels: (N,) or None
        """
        self.request_emb = torch.tensor(request_emb, dtype=torch.float32)
        self.history_emb = torch.tensor(history_emb, dtype=torch.float32)
        self.metadata = torch.tensor(metadata, dtype=torch.float32)
        self.labels = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )

    def __len__(self):
        return len(self.request_emb)

    def __getitem__(self, idx):
        item = {
            "request": self.request_emb[idx],
            "history": self.history_emb[idx],
            "metadata": self.metadata[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class GatedAttentionMLP(nn.Module):
    def __init__(self, sbert_dim, meta_dim, hidden_dim, dropout):
        super(GatedAttentionMLP, self).__init__()

        # 1. Attention Mechanism (No learned parameters for simple dot-product,
        #    but we can add a scaling factor or projection if needed.
        #    Here we stick to simple dot-product as per design).
        self.sbert_dim = sbert_dim

        # 2. Metadata Branch (Credibility Gate)
        # Projects metadata to the size of the concatenated semantic vector (Request + Context)
        # Concatenated size = sbert_dim * 2
        self.meta_gate_net = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, sbert_dim * 2),
            nn.Sigmoid(),  # Gate activation
        )

        # 3. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(sbert_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout_emb = nn.Dropout(dropout)

    def forward(self, request, history, metadata):
        # request: (Batch, 384)
        # history: (Batch, SeqLen, 384)
        # metadata: (Batch, MetaDim)

        # --- Branch 1 & 2: Attention ---
        # Query: Request, Key/Value: History

        # Prepare Query: (Batch, 1, 384)
        query = request.unsqueeze(1)

        # Calculate Scores: (Batch, 1, SeqLen)
        # Q * K^T
        scores = torch.bmm(query, history.transpose(1, 2))

        # Scale scores
        scores = scores / (self.sbert_dim**0.5)

        # Attention Weights: (Batch, 1, SeqLen)
        # Masking zero-padding could be done here if we had lengths,
        # but 0-vectors in history yield 0 dot product (before softmax).
        # Softmax will distribute probability.
        # Ideally we mask, but for this simplified implementation,
        # we assume SBERT 0-vectors are handled reasonably or noise is low.
        attn_weights = torch.softmax(scores, dim=-1)

        # Context Vector: (Batch, 1, 384) -> (Batch, 384)
        context = torch.bmm(attn_weights, history).squeeze(1)

        # Apply dropout to embeddings
        request_drop = self.dropout_emb(request)
        context_drop = self.dropout_emb(context)

        # Concatenate Semantic Vector: (Batch, 768)
        semantic_vector = torch.cat([request_drop, context_drop], dim=1)

        # --- Branch 3: Metadata Gating ---
        # Gate: (Batch, 768)
        gate = self.meta_gate_net(metadata)

        # --- Fusion ---
        # Gated Semantic Vector
        gated_semantic = semantic_vector * gate

        # --- Classification ---
        logits = self.classifier(gated_semantic)

        return logits.squeeze(1)


# =============================================================================
# STREAM B: MLP CONTROLLER
# =============================================================================
class MLPStream:
    """
    Implements Stream B: Direct-Attention Credibility-Gated MLP.
    """

    def __init__(self):
        self.device = config.DEVICE
        self.cache_path = os.path.join(config.WORKING_DIR, "mlp_data.npz")

        # Set seeds
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.RANDOM_SEED)

    def _prepare_data(self, load_cached_data=True):
        """
        Loads features, preprocesses metadata, and caches tensors.
        """
        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(self.cache_path):
            print("Loading MLP data from cache...")
            try:
                data = np.load(self.cache_path)
                return (
                    data["req_train"],
                    data["req_val"],
                    data["req_test"],
                    data["hist_train"],
                    data["hist_val"],
                    data["hist_test"],
                    data["meta_train"],
                    data["meta_val"],
                    data["meta_test"],
                    data["y_train"],
                    data["y_val"],
                )
            except Exception as e:
                print(f"Failed to load MLP cache: {e}. Regenerating...")

        print("Preparing MLP data from scratch...")

        # 2. Load Targets
        df_train, df_val, df_test = data_loader.load_tabular_data(
            load_cached_data=load_cached_data
        )

        if config.TARGET_COL in df_train.columns:
            y_train = df_train[config.TARGET_COL].values.astype(np.float32)
        else:
            raise ValueError("Target column missing in train.")

        if config.TARGET_COL in df_val.columns:
            y_val = df_val[config.TARGET_COL].values.astype(np.float32)
        else:
            raise ValueError("Target column missing in val.")

        # 3. Load SBERT Embeddings
        sbert_enc = text_processing.SbertEncoder()
        req_emb = sbert_enc.generate_request_embeddings(
            df_train, df_val, df_test, load_cached_data=load_cached_data
        )
        hist_emb = sbert_enc.generate_history_embeddings(
            df_train, df_val, df_test, load_cached_data=load_cached_data
        )

        # 4. Load Metadata
        X_tab_train, X_tab_val, X_tab_test = feature_engineering.generate_features(
            load_cached_data=load_cached_data
        )

        # Select numeric columns (including engineered ones)
        # We use all columns generated by feature_engineering except purely string ones if any remain
        # Ideally, feature_engineering returns numeric-ready dataframes (TE + Meta)
        # We ensure we select only numeric types

        def get_numeric_matrix(df):
            return (
                df.select_dtypes(include=[np.number])
                .fillna(0)
                .values.astype(np.float32)
            )

        meta_train = get_numeric_matrix(X_tab_train)
        meta_val = get_numeric_matrix(X_tab_val)
        meta_test = get_numeric_matrix(X_tab_test)

        # 5. Preprocess Metadata (Arcsinh + StandardScaler)
        print("Preprocessing metadata (Arcsinh + StandardScaler)...")
        if config.MLP_USE_ARCSINH:
            meta_train = np.arcsinh(meta_train)
            meta_val = np.arcsinh(meta_val)
            meta_test = np.arcsinh(meta_test)

        scaler = StandardScaler()
        meta_train = scaler.fit_transform(meta_train)
        meta_val = scaler.transform(meta_val)
        meta_test = scaler.transform(meta_test)

        # 6. Save to Cache
        print(f"Saving MLP data to {self.cache_path}...")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez(
            self.cache_path,
            req_train=req_emb["train"],
            req_val=req_emb["val"],
            req_test=req_emb["test"],
            hist_train=hist_emb["train"],
            hist_val=hist_emb["val"],
            hist_test=hist_emb["test"],
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            y_train=y_train,
            y_val=y_val,
        )

        return (
            req_emb["train"],
            req_emb["val"],
            req_emb["test"],
            hist_emb["train"],
            hist_emb["val"],
            hist_emb["test"],
            meta_train,
            meta_val,
            meta_test,
            y_train,
            y_val,
        )

    def train(
        self,
        req_train,
        hist_train,
        meta_train,
        y_train,
        req_val,
        hist_val,
        meta_val,
        y_val,
    ):

        # Hyperparameters
        params = config.MLP_TRAIN_PARAMS
        batch_size = params["batch_size"]
        lr = params["learning_rate"]
        epochs = params["epochs"]
        patience = params["patience"]

        # Datasets
        train_dataset = PizzaDataset(req_train, hist_train, meta_train, y_train)
        val_dataset = PizzaDataset(req_val, hist_val, meta_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Model
        sbert_dim = req_train.shape[1]
        meta_dim = meta_train.shape[1]

        model = GatedAttentionMLP(
            sbert_dim=sbert_dim,
            meta_dim=meta_dim,
            hidden_dim=config.MLP_HIDDEN_DIM,
            dropout=config.MLP_DROPOUT,
        ).to(self.device)

        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=params["weight_decay"]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=params["scheduler_factor"],
            patience=params["scheduler_patience"],
            verbose=True,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_auc = 0.0
        patience_counter = 0
        best_model_state = None

        print(f"Starting MLP training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                req = batch["request"].to(self.device)
                hist = batch["history"].to(self.device)
                meta = batch["metadata"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = model(req, hist, meta)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * req.size(0)

            train_loss /= len(train_dataset)

            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    req = batch["request"].to(self.device)
                    hist = batch["history"].to(self.device)
                    meta = batch["metadata"].to(self.device)
                    labels = batch["label"].to(self.device)

                    outputs = model(req, hist, meta)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * req.size(0)

                    probs = torch.sigmoid(outputs)
                    val_preds.extend(probs.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            val_loss /= len(val_dataset)
            val_auc = roc_auc_score(val_targets, val_preds)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc}"
            )

            scheduler.step(val_auc)

            # Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, best_auc

    def run(self, load_cached_data=True):
        """
        Executes the MLP pipeline.
        """
        # 1. Prepare Data
        (
            req_train,
            req_val,
            req_test,
            hist_train,
            hist_val,
            hist_test,
            meta_train,
            meta_val,
            meta_test,
            y_train,
            y_val,
        ) = self._prepare_data(load_cached_data=load_cached_data)

        # 2. Train
        model, best_auc = self.train(
            req_train,
            hist_train,
            meta_train,
            y_train,
            req_val,
            hist_val,
            meta_val,
            y_val,
        )
        print(f"MLP Best Validation AUC: {best_auc}")

        # 3. Inference
        model.eval()

        def predict(req, hist, meta):
            dataset = PizzaDataset(req, hist, meta)
            loader = DataLoader(
                dataset, batch_size=config.MLP_TRAIN_PARAMS["batch_size"], shuffle=False
            )
            preds = []
            with torch.no_grad():
                for batch in loader:
                    r = batch["request"].to(self.device)
                    h = batch["history"].to(self.device)
                    m = batch["metadata"].to(self.device)
                    out = model(r, h, m)
                    preds.extend(torch.sigmoid(out).cpu().numpy())
            return np.array(preds)

        print("Generating predictions...")
        val_probs = predict(req_val, hist_val, meta_val)
        test_probs = predict(req_test, hist_test, meta_test)

        return val_probs, test_probs, model
