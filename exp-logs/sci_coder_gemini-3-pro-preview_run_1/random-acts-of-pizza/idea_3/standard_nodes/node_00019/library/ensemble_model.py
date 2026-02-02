import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import (
    RF_ESTIMATORS,
    RF_CLASS_WEIGHT,
    RF_N_JOBS,
    MLP_HIDDEN_DIM,
    MLP_DROPOUT,
    MLP_LR,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_PATIENCE,
    FUSION_WEIGHT_RF,
    FUSION_WEIGHT_LR,
    RANDOM_SEED,
)


# Cite solution_lesson_node_00012: Dual-Branch Architectures for Multimodal Feature Fusion
class DualBranchMLP(nn.Module):
    def __init__(self, embedding_dim, meta_dim, hidden_dim, dropout):
        super(DualBranchMLP, self).__init__()

        # Branch A: Text Embeddings
        # Cite solution_lesson_node_00013: Batch Normalization layers
        self.branch_a = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Branch B: Metadata
        self.branch_b = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_emb, x_meta):
        out_a = self.branch_a(x_emb)
        out_b = self.branch_b(x_meta)
        combined = torch.cat([out_a, out_b], dim=1)
        return torch.sigmoid(self.fusion(combined))


class MLPClassifier:
    def __init__(
        self,
        embedding_dim=384,
        hidden_dim=MLP_HIDDEN_DIM,
        dropout=MLP_DROPOUT,
        lr=MLP_LR,
        batch_size=MLP_BATCH_SIZE,
        epochs=MLP_EPOCHS,
        patience=MLP_PATIENCE,
        random_state=RANDOM_SEED,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def fit(self, X, y, X_val=None, y_val=None):
        # Split X into embeddings and metadata
        # Assuming X is [embeddings (384) | metadata (N)]
        meta_dim = X.shape[1] - self.embedding_dim

        self.model = DualBranchMLP(
            self.embedding_dim, meta_dim, self.hidden_dim, self.dropout
        ).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        # Prepare DataLoaders
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_auc = 0
        patience_counter = 0

        print(f"Training MLP on device: {self.device}")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # Split input
                x_emb = batch_X[:, : self.embedding_dim]
                x_meta = batch_X[:, self.embedding_dim :]

                optimizer.zero_grad()
                outputs = self.model(x_emb, x_meta)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            if X_val is not None and y_val is not None:
                val_auc = self.score(X_val, y_val)
                if val_auc > best_auc:
                    best_auc = val_auc
                    patience_counter = 0
                    # Save best state
                    best_weights = self.model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.model.load_state_dict(best_weights)
                    break

        if X_val is not None and y_val is not None and "best_weights" in locals():
            self.model.load_state_dict(best_weights)

        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Split input
        x_emb = X_tensor[:, : self.embedding_dim]
        x_meta = X_tensor[:, self.embedding_dim :]

        with torch.no_grad():
            preds = self.model(x_emb, x_meta).cpu().numpy()

        # Return in sklearn format (N, 2) where col 1 is positive class
        return np.hstack([1 - preds, preds])

    def score(self, X, y):
        probs = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, probs)


class HybridEnsemble:
    """
    A Hybrid Ensemble model that combines a Random Forest (Sparse Stream)
    and a Dual-Branch MLP (Dense Stream) via weighted averaging.
    """

    def __init__(
        self,
        rf_estimators=RF_ESTIMATORS,
        rf_class_weight=RF_CLASS_WEIGHT,
        rf_n_jobs=RF_N_JOBS,
        fusion_weight_rf=FUSION_WEIGHT_RF,
        fusion_weight_lr=FUSION_WEIGHT_LR,
        random_state=RANDOM_SEED,
    ):
        """
        Initializes the ensemble with specific hyperparameters for both branches.
        """
        self.random_state = random_state
        self.fusion_weight_rf = fusion_weight_rf
        self.fusion_weight_lr = fusion_weight_lr

        # Initialize Sparse-Tabular Learner (Random Forest)
        self.rf_model = RandomForestClassifier(
            n_estimators=rf_estimators,
            class_weight=rf_class_weight,
            n_jobs=rf_n_jobs,
            random_state=self.random_state,
            verbose=0,
        )

        # Initialize Dense-Semantic Learner (Dual-Branch MLP)
        self.mlp_model = MLPClassifier(random_state=self.random_state)

    def fit(
        self,
        X_sparse_train,
        X_dense_train,
        y_train,
        X_sparse_val=None,
        X_dense_val=None,
        y_val=None,
    ):
        """
        Trains both models on their respective data streams.
        Optionally evaluates on validation data if provided.
        """
        print("Training Random Forest on Sparse Stream...")
        self.rf_model.fit(X_sparse_train, y_train)

        print("Training Dual-Branch MLP on Dense Stream...")
        self.mlp_model.fit(X_dense_train, y_train, X_dense_val, y_val)

        # Evaluation if validation data is provided
        if X_sparse_val is not None and X_dense_val is not None and y_val is not None:
            print("Evaluating model on validation set...")

            # Get individual predictions
            rf_probs = self.rf_model.predict_proba(X_sparse_val)[:, 1]
            mlp_probs = self.mlp_model.predict_proba(X_dense_val)[:, 1]

            # Calculate individual metrics
            rf_auc = roc_auc_score(y_val, rf_probs)
            mlp_auc = roc_auc_score(y_val, mlp_probs)

            # Get fused predictions
            fused_probs = (
                self.fusion_weight_rf * rf_probs + self.fusion_weight_lr * mlp_probs
            )
            fused_auc = roc_auc_score(y_val, fused_probs)

            print(f"Random Forest Validation AUC: {rf_auc}")
            print(f"Dual-Branch MLP Validation AUC: {mlp_auc}")
            print(f"Hybrid Ensemble Validation AUC: {fused_auc}")

        return self

    def predict_proba(self, X_sparse, X_dense):
        """
        Generates fused probability predictions.

        Args:
            X_sparse: Sparse matrix for RF
            X_dense: Dense array for MLP

        Returns:
            np.array: Probability of the positive class (1).
        """
        # Get probabilities for positive class (index 1)
        rf_probs = self.rf_model.predict_proba(X_sparse)[:, 1]
        mlp_probs = self.mlp_model.predict_proba(X_dense)[:, 1]

        # Weighted Average Fusion
        final_probs = (
            self.fusion_weight_rf * rf_probs + self.fusion_weight_lr * mlp_probs
        )

        return final_probs
