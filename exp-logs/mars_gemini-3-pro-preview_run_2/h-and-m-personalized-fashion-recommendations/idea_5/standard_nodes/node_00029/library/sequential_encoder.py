import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import time
from pathlib import Path
from library import config
from library import utils

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================


class Item2Vec(nn.Module):
    """
    Simple Item2Vec model using Skip-gram with Negative Sampling.
    Cite Lesson 00027: Prefer Mean-Pooled Item Embeddings over Sequential Encoders.
    """

    def __init__(self, vocab_size, params):
        super(Item2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = params["embedding_dim"]

        # Input embedding (Target)
        self.in_embed = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        # Output embedding (Context)
        self.out_embed = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)

        # Init
        self.in_embed.weight.data.uniform_(
            -0.5 / self.embedding_dim, 0.5 / self.embedding_dim
        )
        self.out_embed.weight.data.uniform_(
            -0.5 / self.embedding_dim, 0.5 / self.embedding_dim
        )

    def forward(self, target_ids, context_ids, negative_ids):
        # target_ids: (Batch,)
        # context_ids: (Batch,)
        # negative_ids: (Batch, n_neg)

        # (Batch, Dim)
        u = self.in_embed(target_ids)
        # (Batch, Dim)
        v = self.out_embed(context_ids)
        # (Batch, n_neg, Dim)
        neg = self.out_embed(negative_ids)

        # Positive Score: u dot v -> (Batch,)
        pos_score = torch.sum(u * v, dim=1)

        # Negative Score: u dot neg -> (Batch, n_neg)
        # u.unsqueeze(1): (Batch, 1, Dim)
        # neg.transpose(1, 2): (Batch, Dim, n_neg)
        # bmm -> (Batch, 1, n_neg) -> squeeze -> (Batch, n_neg)
        neg_score = torch.bmm(neg, u.unsqueeze(2)).squeeze(2)

        return pos_score, neg_score


# ==========================================
# 2. DATASET
# ==========================================


class Item2VecDataset(Dataset):
    def __init__(self, sequences, vocab_size, window_size=3, negatives=5):
        self.sequences = sequences
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.negatives = negatives

        # Flatten sequences to list of valid items for faster sampling
        # We only keep sequences that have at least 2 items
        self.valid_indices = [
            i for i, seq in enumerate(sequences) if (seq != 0).sum() > 1
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        seq = self.sequences[real_idx]

        # Filter padding
        items = seq[seq != 0]
        length = len(items)

        # Pick target
        target_idx = np.random.randint(0, length)
        target_item = items[target_idx]

        # Pick context
        start = max(0, target_idx - self.window_size)
        end = min(length, target_idx + self.window_size + 1)
        window_indices = list(range(start, end))
        window_indices.remove(target_idx)

        if not window_indices:
            context_item = items[target_idx]
        else:
            context_item = items[np.random.choice(window_indices)]

        # Pick negatives
        neg_items = np.random.randint(1, self.vocab_size, size=self.negatives)

        return target_item, context_item, neg_items


# ==========================================
# 3. TRAINING FUNCTION
# ==========================================


def train_sequential_model(data_dict, params=None, load_cached_data=False):
    """
    Trains the Item2Vec model or loads it from cache.
    """
    if params is None:
        params = config.SEQ_CONFIG

    model_path = config.SEQ_MODEL_PATH

    # 1. Check Cache
    if load_cached_data and model_path.exists():
        print(f"Loading cached Item2Vec Model from {model_path}...")
        try:
            vocab_size = data_dict["vocab_size"]
            model = Item2Vec(vocab_size, params)
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            model.to(config.DEVICE)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model cache: {e}. Retraining...")

    # 2. Setup Data
    print("Setting up Item2Vec training...")
    sequences = data_dict["sequences"]
    vocab_size = data_dict["vocab_size"]

    dataset = Item2VecDataset(
        sequences,
        vocab_size,
        window_size=params.get("window_size", 3),
        negatives=params.get("negatives", 5),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Setup Model
    utils.seed_everything(config.RANDOM_STATE)
    model = Item2Vec(vocab_size, params).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    # 4. Training Loop
    print(f"Starting training on {config.DEVICE}...")

    for epoch in range(params["epochs"]):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        for batch_idx, (target, context, negatives) in enumerate(dataloader):
            target = target.to(config.DEVICE)
            context = context.to(config.DEVICE)
            negatives = negatives.to(config.DEVICE)

            optimizer.zero_grad()

            pos_score, neg_score = model(target, context, negatives)

            # Loss: -log(sigmoid(pos)) - sum(log(sigmoid(-neg)))
            loss = -torch.mean(
                torch.nn.functional.logsigmoid(pos_score)
                + torch.sum(torch.nn.functional.logsigmoid(-neg_score), dim=1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{params['epochs']} | Time: {elapsed:.1f}s | Loss: {avg_loss:.4f}"
        )

    # Save
    torch.save(model.state_dict(), model_path)
    model.eval()
    return model


# ==========================================
# 4. EMBEDDING EXTRACTION
# ==========================================


def extract_embeddings(model, data_dict, batch_size=4096):
    """
    Extracts item embeddings and computes user embeddings via Mean Pooling.
    Cite Lesson 00027: Prefer Mean-Pooled Item Embeddings.
    """
    print("Extracting embeddings from Item2Vec Model...")

    model.eval()
    # Extract Item Embeddings (Input Embeddings)
    item_embeddings = model.in_embed.weight.detach().cpu().numpy()

    # Compute User Embeddings (Mean of History)
    sequences = data_dict["sequences"]

    print("Computing User Embeddings (Mean Pooling)...")

    user_embeddings_list = []
    loader = DataLoader(
        sequences, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS
    )

    # Move item embeddings to GPU for gathering
    item_emb_tensor = model.in_embed.weight.detach()

    with torch.no_grad():
        for batch_seq in loader:
            batch_seq = batch_seq.to(config.DEVICE)

            # Mask padding (0)
            mask = (batch_seq != 0).float().unsqueeze(2)  # (Batch, Seq, 1)

            # Gather embeddings
            # (Batch, Seq) -> (Batch, Seq, Dim)
            embs = item_emb_tensor[batch_seq]

            # Sum and Count
            sum_embs = torch.sum(embs * mask, dim=1)  # (Batch, Dim)
            counts = torch.sum(mask, dim=1)  # (Batch, 1)

            # Avoid div by zero
            counts = torch.clamp(counts, min=1.0)

            mean_embs = sum_embs / counts
            user_embeddings_list.append(mean_embs.cpu().numpy())

    user_embeddings = np.concatenate(user_embeddings_list, axis=0)

    print(f"Extracted User Embeddings: {user_embeddings.shape}")
    print(f"Extracted Item Embeddings: {item_embeddings.shape}")

    return user_embeddings, item_embeddings
