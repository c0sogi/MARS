import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.data_utils import clean_and_tokenize, identify_identity_indices
from library.metrics import calculate_jigsaw_metrics


# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NBOWModel(nn.Module):
    """
    Neural Bag-of-Words (NBOW) model for toxicity classification.
    Uses an EmbeddingBag layer to average word vectors, followed by an MLP.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout_rate):
        super(NBOWModel, self).__init__()

        # EmbeddingBag efficiently computes the mean of embeddings for a sequence
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")

        # MLP layers
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Output layer
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, offsets):
        """
        Args:
            text (Tensor): 1D tensor containing concatenated indices of all examples in the batch.
            offsets (Tensor): 1D tensor containing the starting index of each example in 'text'.
        """
        embedded = self.embedding(text, offsets)
        x = self.fc1(embedded)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ToxicityDataset(Dataset):
    """
    Dataset class for toxicity text data.
    Implements Stochastic Identity Masking during training to mitigate bias.
    """

    def __init__(
        self,
        texts,
        targets=None,
        vocab=None,
        identity_indices=None,
        mask_prob=0.0,
        is_training=False,
    ):
        self.texts = texts
        self.targets = targets
        self.vocab = vocab
        self.identity_indices = (
            identity_indices if identity_indices is not None else set()
        )
        self.mask_prob = mask_prob
        self.is_training = is_training

        # Pre-fetch special tokens
        self.mask_index = vocab.get_mask_index() if vocab else 0

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = clean_and_tokenize(text)
        indices = self.vocab.lookup_indices(tokens)

        # Stochastic Identity Masking
        if self.is_training and self.mask_prob > 0 and self.identity_indices:
            masked_indices = []
            for i in indices:
                if i in self.identity_indices and random.random() < self.mask_prob:
                    masked_indices.append(self.mask_index)
                else:
                    masked_indices.append(i)
            indices = masked_indices

        # Handle empty sequences
        if len(indices) == 0:
            indices = [
                0
            ]  # Use PAD or UNK index if empty, assuming 0 is safe or handled

        indices_tensor = torch.tensor(indices, dtype=torch.long)

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return indices_tensor, target
        else:
            return indices_tensor


def collate_batch(batch):
    """
    Collate function for EmbeddingBag.
    Concatenates indices into a single 1D tensor and computes offsets.
    """
    label_list, text_list, offsets = [], [], [0]

    has_labels = isinstance(batch[0], tuple)

    if has_labels:
        for _text, _label in batch:
            label_list.append(_label)
            text_list.append(_text)
            offsets.append(_text.size(0))
    else:
        for _text in batch:
            text_list.append(_text)
            offsets.append(_text.size(0))

    # The offsets are cumulative sums of lengths (starting at 0)
    # The last offset is not needed for the start positions, but torch.cumsum includes it.
    # We need [0, len1, len1+len2, ...] up to the last element.
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)

    if has_labels:
        label_list = torch.stack(label_list)
        return text_list, offsets, label_list
    else:
        return text_list, offsets


def train_model(
    train_df,
    val_df,
    vocab,
    batch_size=Config.BATCH_SIZE,
    epochs=Config.NUM_EPOCHS,
    lr=Config.LEARNING_RATE,
    patience=Config.PATIENCE,
    device=None,
):
    """
    Trains the NBOW model with early stopping based on the competition metric.
    """
    set_seed(Config.SEED)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on device: {device}")

    # Identify identity terms for masking
    identity_indices = identify_identity_indices(vocab)

    # Prepare Datasets
    train_dataset = ToxicityDataset(
        texts=train_df[Config.TEXT_COL].tolist(),
        targets=train_df[Config.TARGET_COL].tolist(),
        vocab=vocab,
        identity_indices=identity_indices,
        mask_prob=Config.IDENTITY_MASK_PROB,
        is_training=True,
    )

    val_dataset = ToxicityDataset(
        texts=val_df[Config.TEXT_COL].tolist(),
        targets=val_df[Config.TARGET_COL].tolist(),
        vocab=vocab,
        identity_indices=identity_indices,
        mask_prob=0.0,  # No masking during validation
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    # Initialize Model
    model = NBOWModel(
        vocab_size=len(vocab),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout_rate=Config.DROPOUT,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_score = -float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for texts, offsets, targets in train_loader:
            texts, offsets, targets = (
                texts.to(device),
                offsets.to(device),
                targets.to(device),
            )

            optimizer.zero_grad()
            outputs = model(texts, offsets).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)

        avg_train_loss = train_loss / len(train_dataset)

        # --- Validation ---
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for texts, offsets, targets in val_loader:
                texts, offsets, targets = (
                    texts.to(device),
                    offsets.to(device),
                    targets.to(device),
                )
                outputs = model(texts, offsets).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)

                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataset)

        # --- Metrics Calculation ---
        # We need to pass the predictions back to the dataframe structure to use the library metric function
        val_df_eval = val_df.copy()
        val_df_eval["prediction"] = val_preds

        metrics = calculate_jigsaw_metrics(val_df_eval, prediction_col="prediction")
        final_score = metrics["final_score"]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Score: {final_score:.10f} | "
            f"Overall AUC: {metrics['overall_auc']:.6f} | "
            f"Bias AUCs (Sub/BPSN/BNSP): {metrics['subgroup_auc_mean']:.4f}/{metrics['bpsn_auc_mean']:.4f}/{metrics['bnsp_auc_mean']:.4f}"
        )

        # --- Early Stopping ---
        if final_score > best_score:
            best_score = final_score
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save best model to disk immediately to ensure it's available
            torch.save(best_model_state, Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict_model(model, test_df, vocab, batch_size=Config.BATCH_SIZE, device=None):
    """
    Generates predictions for the test set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    dataset = ToxicityDataset(
        texts=test_df[Config.TEXT_COL].tolist(),
        targets=None,
        vocab=vocab,
        is_training=False,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # Unpack based on collate_batch return (text, offsets) for inference
            texts, offsets = batch
            texts, offsets = texts.to(device), offsets.to(device)

            outputs = model(texts, offsets).squeeze()

            # Handle single-item batch edge case where squeeze might remove batch dim
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)

            all_preds.extend(outputs.cpu().numpy())

    # Create submission dataframe
    submission = pd.DataFrame(
        {Config.ID_COL: test_df[Config.ID_COL], "prediction": all_preds}
    )

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    return submission
