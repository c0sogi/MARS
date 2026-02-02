import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import EssayDataset
from library.metrics import compute_qwk


def _kappa_loss(coef, X, y):
    """
    Loss function for threshold optimization (Negative QWK).
    """
    X_p = np.digitize(X, coef) + 1
    return -cohen_kappa_score(y, X_p, weights="quadratic")


class EssayModel(nn.Module):
    """
    Transformer model for Essay Scoring with Mean+Max Pooling.
    Cite Lesson 00011: Robust Regression Heads.
    """

    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Mean + Max Pooling concatenates two vectors of hidden_size
        self.pooler_dim = self.config.hidden_size * 2

        self.fc = nn.Linear(self.pooler_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (BS, SeqLen, Hidden)

        # Masking for pooling
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Mean Pooling
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # Max Pooling
        # Set masked values to large negative number so they don't affect max
        last_hidden_state[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(last_hidden_state, 1)[0]

        # Concatenate
        pooled_output = torch.cat([mean_embeddings, max_embeddings], 1)

        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x


class ScoreRegressor:
    """
    Wrapper for PyTorch Transformer training and inference.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = EssayModel(Config.MODEL_NAME).to(self.device)
        self.thresholds = None

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the Transformer model.
        Cite Lesson 00010: Uniform Learning Rate.
        Cite Lesson 00011: SmoothL1Loss.
        """
        # Create Datasets
        train_ds = EssayDataset(X_train, y_train, self.tokenizer, Config.MAX_LENGTH)
        val_ds = EssayDataset(X_val, y_val, self.tokenizer, Config.MAX_LENGTH)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.SmoothL1Loss()

        best_loss = float("inf")
        best_state = None

        print(f"Starting training on {self.device}...")

        for epoch in range(Config.EPOCHS):
            self.model.train()
            train_loss = 0

            for batch in train_loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                targets = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(ids, mask).squeeze(-1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            val_preds = []

            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(self.device)
                    mask = batch["attention_mask"].to(self.device)
                    targets = batch["labels"].to(self.device)

                    outputs = self.model(ids, mask).squeeze(-1)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_preds.extend(outputs.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
            )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_state = self.model.state_dict()

        # Load best model
        if best_state:
            self.model.load_state_dict(best_state)

        # Optimize Thresholds (Cite Lesson 00002)
        print("Optimizing thresholds...")
        self.model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                outputs = self.model(ids, mask).squeeze(-1)
                val_preds.extend(outputs.cpu().numpy())

        val_preds = np.array(val_preds)
        initial_coef = [1.5, 2.5, 3.5, 4.5, 5.5]
        opt_res = minimize(
            _kappa_loss,
            initial_coef,
            args=(val_preds, y_val),
            method="nelder-mead",
            tol=1e-3,
        )
        self.thresholds = opt_res.x
        print(f"Optimized Thresholds: {self.thresholds}")
        print(f"Best Val QWK: {-opt_res.fun}")

        return self

    def predict(self, X):
        """
        Inference method.
        """
        ds = EssayDataset(X, None, self.tokenizer, Config.MAX_LENGTH)
        loader = DataLoader(
            ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                outputs = self.model(ids, mask).squeeze(-1)
                preds.extend(outputs.cpu().numpy())

        preds = np.array(preds)

        if self.thresholds is not None:
            return np.digitize(preds, self.thresholds) + 1
        else:
            return np.clip(preds, 1, 6)

    def save(self, path):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "thresholds": self.thresholds,
                "config": Config.MODEL_NAME,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location=Config.DEVICE)
        instance = cls()
        instance.model.load_state_dict(checkpoint["model_state"])
        instance.thresholds = checkpoint["thresholds"]
        return instance
