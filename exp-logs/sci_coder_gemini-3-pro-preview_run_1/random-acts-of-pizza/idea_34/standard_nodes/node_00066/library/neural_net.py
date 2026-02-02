import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Pizza Request data.
    Handles Title, Body, History embeddings, and Metadata.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Output dictionary from FeaturePipeline.
        """
        self.title_emb = data_dict["mlp_title_emb"]
        self.body_emb = data_dict["mlp_body_emb"]
        self.history_emb = data_dict["mlp_history_emb"]
        self.metadata = data_dict["mlp_metadata"]

        if "labels" in data_dict:
            self.labels = data_dict["labels"]
        else:
            self.labels = None

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        # Retrieve items
        t_emb = self.title_emb[idx]
        b_emb = self.body_emb[idx]
        h_emb = self.history_emb[idx]
        meta = self.metadata[idx]

        # Convert to torch tensors
        t_tensor = torch.tensor(t_emb, dtype=torch.float32)
        b_tensor = torch.tensor(b_emb, dtype=torch.float32)
        meta_tensor = torch.tensor(meta, dtype=torch.float32)

        # History is variable length (N, 384), keep as tensor for collate
        # Handle case where history might be empty (0, 384)
        if h_emb.shape[0] == 0:
            # Create a dummy history of shape (1, 384) of zeros to avoid dimension errors,
            # but mask it out later.
            h_tensor = torch.zeros((1, 384), dtype=torch.float32)
            h_len = 0
        else:
            h_tensor = torch.tensor(h_emb, dtype=torch.float32)
            h_len = h_emb.shape[0]

        item = {
            "title": t_tensor,
            "body": b_tensor,
            "history": h_tensor,
            "history_len": h_len,
            "metadata": meta_tensor,
        }

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable length history sequences.
        Pads history to the max length in the batch.
        """
        titles = torch.stack([item["title"] for item in batch])
        bodies = torch.stack([item["body"] for item in batch])
        metadata = torch.stack([item["metadata"] for item in batch])

        # Handle History Padding
        histories = [item["history"] for item in batch]
        lengths = [item["history_len"] for item in batch]
        max_len = max(max(lengths), 1)  # Ensure at least 1 for dimension sanity

        emb_dim = histories[0].shape[1]
        batch_size = len(batch)

        padded_history = torch.zeros(batch_size, max_len, emb_dim, dtype=torch.float32)
        mask = torch.zeros(
            batch_size, max_len, dtype=torch.bool
        )  # 1/True for padding, 0/False for valid (or vice versa)
        # We will use additive mask: 0 for valid, -inf for padding.
        # So let's create a mask where 1 is valid, 0 is padding first.
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

        for i, (h, l) in enumerate(zip(histories, lengths)):
            if l > 0:
                padded_history[i, :l, :] = h
                attention_mask[i, :l] = 1.0
            # If l == 0, it remains zeros, mask remains zeros

        batch_out = {
            "title": titles,
            "body": bodies,
            "history": padded_history,
            "attention_mask": attention_mask,
            "metadata": metadata,
        }

        if "label" in batch[0]:
            batch_out["label"] = torch.stack([item["label"] for item in batch])

        return batch_out


class DualQueryMLP(nn.Module):
    """
    Dropout-Stabilized Dual-Query MLP.

    Architecture:
    1. Title & Body Branches (Raw SBERT)
    2. Dual-Query Attention (Title->History, Body->History)
    3. Metadata Branch -> Credibility Gate
    4. Fusion -> MLP -> Output
    """

    def __init__(self, metadata_dim, embedding_dim=384):
        super(DualQueryMLP, self).__init__()

        self.embedding_dim = embedding_dim

        # --- Attention Mechanism ---
        # We use raw dot-product, so no learned weights here strictly for the attention calculation
        # unless we wanted to project Q/K/V. The prompt specifies "Raw SBERT embeddings" for history.
        # We will keep it parameter-free for the attention core to adhere to "Dual-Query History Attention"
        # utilizing raw embeddings, but we will use dropout.

        self.emb_dropout = nn.Dropout(Config.MLP_DROPOUT_EMBEDDING)

        # --- Dimensions ---
        # Semantic Vector = Title + Body + Context_Topic + Context_Narrative + 2 Scalars
        self.semantic_dim = (embedding_dim * 4) + 2

        # --- Metadata Branch (Gating Network) ---
        # Projects metadata to the size of the semantic vector to act as a gate
        self.meta_gate = nn.Sequential(
            nn.Linear(metadata_dim, Config.MLP_PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT_DENSE),
            nn.Linear(Config.MLP_PROJECTION_DIM, self.semantic_dim),
            nn.Sigmoid(),  # Output [0, 1] for gating
        )

        # --- Final Classifier ---
        # Input is the gated semantic vector
        self.classifier = nn.Sequential(
            nn.Linear(self.semantic_dim, Config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT_DENSE),
            nn.Linear(Config.MLP_HIDDEN_DIM, Config.MLP_PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT_DENSE),
            nn.Linear(Config.MLP_PROJECTION_DIM, 1),
        )

    def forward(self, title, body, history, attention_mask, metadata):
        """
        Args:
            title: (B, 384)
            body: (B, 384)
            history: (B, L, 384)
            attention_mask: (B, L) - 1.0 for valid, 0.0 for padding
            metadata: (B, MetaDim)
        """
        # Apply dropout to embeddings
        title = self.emb_dropout(title)
        body = self.emb_dropout(body)
        history = self.emb_dropout(history)

        # --- Dual-Query Attention ---

        # Prepare Mask: (B, 1, L)
        # We want -inf where mask is 0
        mask_expanded = attention_mask.unsqueeze(1)  # (B, 1, L)
        additive_mask = (1.0 - mask_expanded) * -1e9

        # Head A: Topic Context (Query=Title)
        # Q: (B, 1, D), K: (B, L, D) -> Scores: (B, 1, L)
        q_title = title.unsqueeze(1)
        scores_topic = torch.bmm(q_title, history.transpose(1, 2))
        scores_topic = scores_topic + additive_mask
        weights_topic = torch.softmax(scores_topic, dim=-1)  # (B, 1, L)
        context_topic = torch.bmm(weights_topic, history).squeeze(1)  # (B, D)

        # Head B: Narrative Context (Query=Body)
        q_body = body.unsqueeze(1)
        scores_narr = torch.bmm(q_body, history.transpose(1, 2))
        scores_narr = scores_narr + additive_mask
        weights_narr = torch.softmax(scores_narr, dim=-1)
        context_narr = torch.bmm(weights_narr, history).squeeze(1)  # (B, D)

        # --- Alignment Injection ---
        # Cosine Similarity between Query and Context
        # Since we just did dot product attention, the context is a weighted sum.
        # We compute dot product (unnormalized cosine if vectors aren't norm'd) or cosine.
        # SBERT embeddings are typically normalized. We'll use CosineSimilarity for robustness.
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        align_topic = cos(title, context_topic).unsqueeze(1)  # (B, 1)
        align_narr = cos(body, context_narr).unsqueeze(1)  # (B, 1)

        # Handle case where history was empty (context is zero vector)
        # If context is zero, cosine might be NaN or 0.
        # Since we padded with zeros and masked, context for empty history is 0.
        # Cosine with 0 vector is undefined/0. We replace NaNs with 0 just in case.
        align_topic = torch.nan_to_num(align_topic, nan=0.0)
        align_narr = torch.nan_to_num(align_narr, nan=0.0)

        # --- Concatenation ---
        # Semantic Vector S
        semantic_vector = torch.cat(
            [title, body, context_topic, context_narr, align_topic, align_narr], dim=1
        )  # (B, 1538)

        # --- Gated Fusion ---
        gate = self.meta_gate(metadata)  # (B, 1538)
        fused_vector = semantic_vector * gate

        # --- Classification ---
        logits = self.classifier(fused_vector)
        return logits.squeeze(1)


class NeuralTrainer:
    """
    Trainer class for the DualQueryMLP.
    """

    def __init__(self, input_dim_metadata):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"NeuralTrainer using device: {self.device}")

        set_seed(Config.RANDOM_STATE)

        self.model = DualQueryMLP(metadata_dim=input_dim_metadata).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.MLP_LEARNING_RATE,
            weight_decay=Config.MLP_WEIGHT_DECAY,
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_val_auc = 0.0
        self.best_model_state = None

    def fit(self, train_data_dict, val_data_dict):
        """
        Trains the model with early stopping.
        """
        # Create Datasets and Loaders
        train_dataset = PizzaDataset(train_data_dict)
        val_dataset = PizzaDataset(val_data_dict)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.MLP_BATCH_SIZE,
            shuffle=True,
            collate_fn=PizzaDataset.collate_fn,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.MLP_BATCH_SIZE,
            shuffle=False,
            collate_fn=PizzaDataset.collate_fn,
            num_workers=0,
        )

        print(f"Starting training for {Config.MLP_EPOCHS} epochs...")
        patience_counter = 0

        for epoch in range(Config.MLP_EPOCHS):
            # --- Training ---
            self.model.train()
            train_loss_sum = 0
            train_preds = []
            train_targets = []

            for batch in train_loader:
                # Move batch to device
                title = batch["title"].to(self.device)
                body = batch["body"].to(self.device)
                history = batch["history"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                meta = batch["metadata"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(title, body, history, mask, meta)
                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()

                train_loss_sum += loss.item() * title.size(0)
                train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
                train_targets.extend(labels.cpu().numpy())

            avg_train_loss = train_loss_sum / len(train_dataset)
            train_auc = roc_auc_score(train_targets, train_preds)

            # --- Validation ---
            self.model.eval()
            val_preds = []
            val_targets = []
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    title = batch["title"].to(self.device)
                    body = batch["body"].to(self.device)
                    history = batch["history"].to(self.device)
                    mask = batch["attention_mask"].to(self.device)
                    meta = batch["metadata"].to(self.device)
                    labels = batch["label"].to(self.device)

                    logits = self.model(title, body, history, mask, meta)
                    loss = self.criterion(logits, labels)

                    val_loss_sum += loss.item() * title.size(0)
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            avg_val_loss = val_loss_sum / len(val_dataset)
            val_auc = roc_auc_score(val_targets, val_preds)

            print(
                f"Epoch {epoch+1}/{Config.MLP_EPOCHS} | "
                f"Train Loss: {avg_train_loss:.6f} | Train AUC: {train_auc:.10f} | "
                f"Val Loss: {avg_val_loss:.6f} | Val AUC: {val_auc:.10f}"
            )

            # --- Early Stopping ---
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
                # Save best model to disk
                torch.save(
                    self.best_model_state,
                    os.path.join(Config.WORKING_DIR, "best_mlp_model.pth"),
                )
            else:
                patience_counter += 1

            if patience_counter >= Config.MLP_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best Val AUC: {self.best_val_auc:.10f}"
                )
                break

        # Load best weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def predict(self, test_data_dict):
        """
        Generates predictions for the test set.
        """
        test_dataset = PizzaDataset(test_data_dict)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.MLP_BATCH_SIZE,
            shuffle=False,
            collate_fn=PizzaDataset.collate_fn,
            num_workers=0,
        )

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                title = batch["title"].to(self.device)
                body = batch["body"].to(self.device)
                history = batch["history"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                meta = batch["metadata"].to(self.device)

                logits = self.model(title, body, history, mask, meta)
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())

        return np.array(all_preds)
