import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import copy

from library.config import MLP_PARAMS, DEVICE, RANDOM_STATE
from library.utils import set_seed


class PizzaDataset(Dataset):
    """
    Custom Dataset to handle dictionary inputs for the MLP.
    """

    def __init__(self, features_dict, targets=None):
        self.features = features_dict
        self.targets = targets
        # Ensure all arrays have the same length
        self.n_samples = len(next(iter(features_dict.values())))
        self.keys = list(features_dict.keys())

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Retrieve sample features
        item = {k: self.features[k][idx] for k in self.keys}

        # Convert to tensor immediately if needed, or rely on collate.
        # Here we rely on default collate which handles numpy arrays -> tensors.

        if self.targets is not None:
            return item, self.targets[idx]
        return item


class DualQueryAttention(nn.Module):
    """
    Computes attention over history using two queries: Title and Body.
    """

    def __init__(self, input_dim, dropout_prob=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout_prob)

        # Projections
        self.W_Q_title = nn.Linear(input_dim, input_dim)
        self.W_Q_body = nn.Linear(input_dim, input_dim)
        self.W_K = nn.Linear(input_dim, input_dim)
        self.W_V = nn.Linear(input_dim, input_dim)

    def forward(self, history_seq, history_mask, query_title, query_body):
        """
        Args:
            history_seq: (B, Seq, D)
            history_mask: (B, Seq) - 1 for valid, 0 for pad
            query_title: (B, D)
            query_body: (B, D)
        Returns:
            context_title, context_body: (B, D)
        """
        B, Seq, D = history_seq.shape

        # Project
        K = self.W_K(history_seq)  # (B, Seq, D)
        V = self.W_V(history_seq)  # (B, Seq, D)
        Q_t = self.W_Q_title(query_title).unsqueeze(1)  # (B, 1, D)
        Q_b = self.W_Q_body(query_body).unsqueeze(1)  # (B, 1, D)

        # Scaled Dot Product Attention
        # Scores: (B, 1, Seq)
        scale = D**0.5
        scores_t = torch.matmul(Q_t, K.transpose(1, 2)) / scale
        scores_b = torch.matmul(Q_b, K.transpose(1, 2)) / scale

        # Masking: Fill pad positions with -inf
        mask_expanded = history_mask.unsqueeze(1)  # (B, 1, Seq)
        scores_t = scores_t.masked_fill(mask_expanded == 0, -1e9)
        scores_b = scores_b.masked_fill(mask_expanded == 0, -1e9)

        # Softmax
        attn_t = F.softmax(scores_t, dim=-1)
        attn_b = F.softmax(scores_b, dim=-1)

        # Dropout
        attn_t = self.dropout(attn_t)
        attn_b = self.dropout(attn_b)

        # Weighted Sum: (B, 1, D) -> (B, D)
        context_t = torch.matmul(attn_t, V).squeeze(1)
        context_b = torch.matmul(attn_b, V).squeeze(1)

        return context_t, context_b


class SkipGatedFusion(nn.Module):
    """
    Gates the semantic vector using dense features, with a skip connection for dense features.
    """

    def __init__(self, semantic_dim, dense_dim, dropout_prob=0.0):
        super().__init__()
        # Gate generator: Dense -> Semantic Dim
        self.gate_fc = nn.Linear(dense_dim, semantic_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, semantic_vector, dense_features):
        """
        Args:
            semantic_vector: (B, S_Dim)
            dense_features: (B, D_Dim)
        Returns:
            fused: (B, S_Dim + D_Dim)
        """
        # Compute Gate (Sigmoid activation)
        gate = torch.sigmoid(self.gate_fc(dense_features))

        # Apply Gate
        gated_semantic = semantic_vector * gate
        gated_semantic = self.dropout(gated_semantic)

        # Concatenate (Skip Connection for Dense Features)
        fused = torch.cat([gated_semantic, dense_features], dim=1)
        return fused


class SkipGatedMLP(nn.Module):
    """
    Main Neural Network Architecture.
    """

    def __init__(
        self,
        input_embedding_dim,
        dense_input_dim,
        hidden_dim,
        dropout_prob=0.5,
        dropout_dense=0.2,
    ):
        super().__init__()

        # 1. Attention Mechanism
        self.attention = DualQueryAttention(input_embedding_dim, dropout_prob)

        # 2. Semantic Vector Construction
        # Components: Title(D) + Body(D) + Ctx_Title(D) + Ctx_Body(D) + Centroid(D)
        self.semantic_dim = input_embedding_dim * 5

        # 3. Fusion Layer
        self.fusion = SkipGatedFusion(self.semantic_dim, dense_input_dim, dropout_dense)

        # 4. Classification Head
        fused_dim = self.semantic_dim + dense_input_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, inputs):
        # Unpack inputs
        title_emb = inputs["title_emb"]
        body_emb = inputs["body_emb"]
        history_emb = inputs["history_emb"]
        history_mask = inputs["history_mask"]
        history_centroid = inputs["history_centroid"]
        dense_features = inputs["dense_features"]

        # Branch 3: Dual Query Attention
        ctx_t, ctx_b = self.attention(history_emb, history_mask, title_emb, body_emb)

        # Construct Semantic Vector
        # [Title, Body, Attended_Contexts, Global_Centroid]
        semantic_vector = torch.cat(
            [title_emb, body_emb, ctx_t, ctx_b, history_centroid], dim=1
        )

        # Branch 5 & Fusion
        fused = self.fusion(semantic_vector, dense_features)

        # Classification
        logits = self.classifier(fused)
        return logits


class MLPPredictor:
    """
    Wrapper class for training and predicting with SkipGatedMLP.
    Follows a scikit-learn like interface.
    """

    def __init__(self, params=None):
        self.params = params if params is not None else MLP_PARAMS.copy()
        self.model = None
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    def _to_tensor_dict(self, batch_dict):
        """Moves a batch dictionary to device."""
        return {k: v.to(self.device) for k, v in batch_dict.items()}

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the MLP model.
        Args:
            X_train (dict): Dictionary of training features.
            y_train (array-like): Training targets.
            X_val (dict, optional): Validation features.
            y_val (array-like, optional): Validation targets.
        """
        set_seed(RANDOM_STATE)

        # Infer dimensions
        input_embedding_dim = X_train["title_emb"].shape[1]
        dense_input_dim = X_train["dense_features"].shape[1]

        # Initialize Model
        self.model = SkipGatedMLP(
            input_embedding_dim=input_embedding_dim,
            dense_input_dim=dense_input_dim,
            hidden_dim=self.params["hidden_dim"],
            dropout_prob=self.params["dropout_prob"],
            dropout_dense=self.params["dropout_dense"],
        ).to(self.device)

        # Prepare DataLoaders
        train_dataset = PizzaDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = PizzaDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.params["batch_size"] * 2,
                shuffle=False,
                num_workers=0,
            )

        # Optimizer & Loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.params["scheduler_factor"],
            patience=self.params["scheduler_patience"],
            verbose=False,
        )

        # Training Loop
        best_val_auc = 0.0
        best_model_state = None
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(self.params["epochs"]):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = self._to_tensor_dict(batch_X)
                batch_y = batch_y.float().to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            if val_loader:
                self.model.eval()
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = self._to_tensor_dict(batch_X)
                        logits = self.model(batch_X)
                        probs = torch.sigmoid(logits)

                        val_preds.extend(probs.cpu().numpy().flatten())
                        val_targets.extend(batch_y.numpy().flatten())

                val_auc = roc_auc_score(val_targets, val_preds)
                scheduler.step(val_auc)

                # Checkpointing
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early Stopping
                if patience_counter >= self.params["patience"]:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                # If no validation set, just save last state
                best_model_state = copy.deepcopy(self.model.state_dict())

        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        print(f"MLP Training Complete. Best Validation AUC: {best_val_auc}")

    def predict(self, X):
        """
        Generates predictions.
        Args:
            X (dict): Dictionary of features.
        Returns:
            np.ndarray: Probabilities of class 1.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        self.model.eval()
        dataset = PizzaDataset(X)
        loader = DataLoader(
            dataset,
            batch_size=self.params["batch_size"] * 2,
            shuffle=False,
            num_workers=0,
        )

        all_probs = []
        with torch.no_grad():
            for batch_X in loader:
                batch_X = self._to_tensor_dict(batch_X)
                logits = self.model(batch_X)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy().flatten())

        return np.array(all_probs)
