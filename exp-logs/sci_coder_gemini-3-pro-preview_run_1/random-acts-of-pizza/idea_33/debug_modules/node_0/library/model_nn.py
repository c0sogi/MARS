import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import copy

from library.config import Config
from library.utils import set_seed, save_torch_model, load_torch_model
from library.dataset import PizzaDataset


class DualQueryAttention(nn.Module):
    """
    Computes Dot-Product Attention between a Query vector and a Sequence of Key/Value vectors.
    Returns the Context vector and the Scalar Similarity between Query and Context.
    """

    def __init__(self, input_dim):
        super(DualQueryAttention, self).__init__()
        # We assume Raw SBERT embeddings, but allow for optional linear projection if needed.
        # Based on description "Processes Raw SBERT... Dot-Product Attention",
        # we keep it simple but add a temperature scaling for stability.
        self.scale = 1.0 / (input_dim**0.5)

    def forward(self, query, history_seq, history_mask):
        """
        Args:
            query: (B, D)
            history_seq: (B, SeqLen, D)
            history_mask: (B, SeqLen) - 1 for valid, 0 for pad
        Returns:
            context: (B, D)
            similarity: (B, 1)
        """
        # Expand query to (B, 1, D) for broadcasting
        query_unsqueezed = query.unsqueeze(1)

        # Compute scores: (B, 1, D) @ (B, D, SeqLen) -> (B, 1, SeqLen)
        scores = torch.bmm(query_unsqueezed, history_seq.transpose(1, 2))
        scores = scores * self.scale

        # Apply Masking
        # Mask is (B, SeqLen). We need (B, 1, SeqLen).
        # Where mask is 0, set score to -inf
        mask_expanded = history_mask.unsqueeze(1)
        scores = scores.masked_fill(mask_expanded == 0, -1e9)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)  # (B, 1, SeqLen)

        # Compute Context: (B, 1, SeqLen) @ (B, SeqLen, D) -> (B, 1, D)
        context = torch.bmm(attn_weights, history_seq)
        context = context.squeeze(1)  # (B, D)

        # Alignment Injection: Scalar similarity between Query and Context
        # (B, D) * (B, D) -> sum -> (B, 1)
        similarity = (query * context).sum(dim=1, keepdim=True)

        return context, similarity


class DualQueryMLP(nn.Module):
    def __init__(self, meta_dim):
        super(DualQueryMLP, self).__init__()

        emb_dim = Config.MLP_INPUT_EMBEDDING_DIM
        hidden_dim = Config.MLP_HIDDEN_DIM

        # --- Branches ---
        # 1 & 2: Title and Body (Raw SBERT)
        # We apply dropout directly to the inputs in the forward pass
        self.emb_dropout = nn.Dropout(Config.MLP_EMBEDDING_DROPOUT)

        # 3: Dual-Query Attention
        self.attention = DualQueryAttention(emb_dim)

        # 4: Metadata Credibility Gate
        # The fusion vector will consist of:
        # Title(D) + Body(D) + Context_Title(D) + Context_Body(D) + Sim_Title(1) + Sim_Body(1)
        self.fusion_dim = (emb_dim * 4) + 2

        self.meta_gate = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DENSE_DROPOUT),
            nn.Linear(hidden_dim, self.fusion_dim),
            nn.Sigmoid(),  # Gate values between 0 and 1
        )

        # --- Prediction Head ---
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DENSE_DROPOUT),
            nn.Linear(hidden_dim, 1),  # Logits
        )

    def forward(self, title_emb, body_emb, history_seq, history_mask, meta):
        # Apply Dropout to embeddings
        t_emb = self.emb_dropout(title_emb)
        b_emb = self.emb_dropout(body_emb)
        h_seq = self.emb_dropout(history_seq)

        # --- Dual-Query Attention ---
        # Head A: Topic Context (Query=Title)
        ctx_title, sim_title = self.attention(t_emb, h_seq, history_mask)

        # Head B: Narrative Context (Query=Body)
        ctx_body, sim_body = self.attention(b_emb, h_seq, history_mask)

        # --- Fusion ---
        # Concatenate all semantic signals
        # Shapes: (B, D), (B, D), (B, D), (B, D), (B, 1), (B, 1)
        fused_semantic = torch.cat(
            [t_emb, b_emb, ctx_title, ctx_body, sim_title, sim_body], dim=1
        )

        # --- Credibility Gate ---
        gate = self.meta_gate(meta)

        # Apply Gate
        gated_features = fused_semantic * gate

        # --- Prediction ---
        logits = self.head(gated_features)
        return logits


class NeuralNetworkStream:
    """
    Manages the MLP stream (Stream B).
    Handles training, early stopping, and inference.
    """

    def __init__(self):
        self.model = None
        self.model_filename = "nn_model.pth"
        self.device = Config.DEVICE

    def train(self, mlp_data, force_retrain=False):
        """
        Trains the DualQueryMLP model.

        Args:
            mlp_data (dict): Dictionary with 'train', 'val' keys containing tensor dicts.
            force_retrain (bool): If True, ignores cached model and retrains.

        Returns:
            float: Best Validation ROC AUC.
            np.ndarray: Validation predictions.
        """
        set_seed()

        # Check for cached model
        if not force_retrain:
            # We need to instantiate the model structure to load weights
            # Determine meta_dim from data
            meta_dim = mlp_data["train"]["meta"].shape[1]
            temp_model = DualQueryMLP(meta_dim).to(self.device)

            if load_torch_model(temp_model, self.model_filename, device=self.device):
                print(f"Loading cached MLP model from {self.model_filename}...")
                self.model = temp_model

                # Generate validation metrics
                val_dataset = PizzaDataset(mlp_data["val"])
                val_loader = DataLoader(
                    val_dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=False
                )
                val_auc, val_preds = self._evaluate(self.model, val_loader)
                print(f"Loaded MLP Validation ROC AUC: {val_auc}")
                return val_auc, val_preds

        print("Training MLP Model...")

        # Prepare Datasets and Loaders
        train_dataset = PizzaDataset(mlp_data["train"])
        val_dataset = PizzaDataset(mlp_data["val"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.MLP_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.MLP_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device == "cuda" else False,
        )

        # Initialize Model
        meta_dim = mlp_data["train"]["meta"].shape[1]
        self.model = DualQueryMLP(meta_dim).to(self.device)

        # Optimizer & Loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.MLP_LEARNING_RATE,
            weight_decay=Config.MLP_WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping Variables
        best_auc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        for epoch in range(Config.MLP_MAX_EPOCHS):
            self.model.train()
            running_loss = 0.0

            for batch in train_loader:
                # Move to device
                title_emb = batch["title_emb"].to(self.device)
                body_emb = batch["body_emb"].to(self.device)
                history_seq = batch["history_seq"].to(self.device)
                history_mask = batch["history_mask"].to(self.device)
                meta = batch["meta"].to(self.device)
                targets = batch["target"].to(self.device).unsqueeze(1)

                optimizer.zero_grad()

                outputs = self.model(
                    title_emb, body_emb, history_seq, history_mask, meta
                )
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * title_emb.size(0)

            epoch_loss = running_loss / len(train_dataset)

            # Validation
            val_auc, _ = self._evaluate(self.model, val_loader)

            print(
                f"Epoch {epoch+1}/{Config.MLP_MAX_EPOCHS} - Loss: {epoch_loss:.4f} - Val AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.MLP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best weights
        self.model.load_state_dict(best_model_wts)
        save_torch_model(self.model, self.model_filename)

        # Final Validation Predictions
        final_auc, final_preds = self._evaluate(self.model, val_loader)
        print(f"Final MLP Validation ROC AUC: {final_auc}")

        return final_auc, final_preds

    def _evaluate(self, model, dataloader):
        """
        Helper to evaluate model on a dataloader.
        """
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                title_emb = batch["title_emb"].to(self.device)
                body_emb = batch["body_emb"].to(self.device)
                history_seq = batch["history_seq"].to(self.device)
                history_mask = batch["history_mask"].to(self.device)
                meta = batch["meta"].to(self.device)

                outputs = model(title_emb, body_emb, history_seq, history_mask, meta)
                probs = torch.sigmoid(outputs).cpu().numpy()

                all_preds.append(probs)
                if "target" in batch:
                    all_targets.append(batch["target"].cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()

        if all_targets:
            all_targets = np.concatenate(all_targets).flatten()
            auc = roc_auc_score(all_targets, all_preds)
            return auc, all_preds
        else:
            return None, all_preds

    def predict(self, mlp_data_test):
        """
        Generates predictions for the test set.

        Args:
            mlp_data_test (dict): Dictionary containing test tensors.

        Returns:
            np.ndarray: Probabilities for class 1.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        test_dataset = PizzaDataset(mlp_data_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.MLP_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device == "cuda" else False,
        )

        _, preds = self._evaluate(self.model, test_loader)
        return preds
