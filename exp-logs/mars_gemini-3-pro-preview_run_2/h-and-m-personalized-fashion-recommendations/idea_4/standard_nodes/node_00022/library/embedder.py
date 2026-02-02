import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import json
import random
from library.config import Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SkipGramModel(nn.Module):
    """
    PyTorch implementation of Skip-gram with Negative Sampling.
    """

    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.u_embeddings = nn.Embedding(vocab_size, embed_dim)  # Target
        self.v_embeddings = nn.Embedding(vocab_size, embed_dim)  # Context

        # Initialize weights
        initrange = 0.5 / embed_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, u_pos, v_pos, v_neg):
        """
        u_pos: [batch_size] - Center words
        v_pos: [batch_size] - Context words
        v_neg: [batch_size, n_neg] - Negative samples
        """
        embed_u = self.u_embeddings(u_pos)  # [batch, dim]
        embed_v = self.v_embeddings(v_pos)  # [batch, dim]

        # Positive score
        score = torch.mul(embed_u, embed_v).sum(dim=1)  # [batch]
        score = torch.nn.functional.logsigmoid(score)

        # Negative score
        neg_embed_v = self.v_embeddings(v_neg)  # [batch, n_neg, dim]
        # bmm: [batch, n_neg, dim] x [batch, dim, 1] -> [batch, n_neg, 1]
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze(2)
        neg_score = torch.nn.functional.logsigmoid(-1 * neg_score)

        # Total loss (negative log likelihood)
        return -1 * (torch.sum(score) + torch.sum(neg_score))


class LatentEmbedder:
    """
    Manages the lifecycle of the Item2Vec model: Data Prep, Training, Caching, and Inference.
    """

    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.embeddings = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(Config.SEED)

    def fit(self, transactions: pd.DataFrame, load_cached_data: bool = True):
        """
        Trains the embedding model or loads it from cache.

        Args:
            transactions: Full transaction history.
            load_cached_data: Whether to attempt loading from disk.
        """
        # Define cache paths based on hyperparameters
        params = {
            "window": Config.W2V_WINDOW,
            "dim": Config.EMBED_DIM,
            "epochs": Config.W2V_EPOCHS,
            "weeks": Config.EMBED_WINDOW_WEEKS,
            "neg": Config.W2V_NEGATIVE,
        }
        vocab_path = Config.get_cache_path("item2vec_vocab.json", params)
        embed_path = Config.get_cache_path("item2vec_embeddings.npy", params)

        # 1. Try to load from cache
        if load_cached_data and vocab_path.exists() and embed_path.exists():
            print(f"Loading cached embeddings from {embed_path}")
            with open(vocab_path, "r") as f:
                self.vocab = json.load(f)
            self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
            self.embeddings = np.load(embed_path)
            return

        # 2. Train from scratch
        print("Training Item2Vec model...")

        # Filter data to the configured window
        max_date = transactions["t_dat"].max()
        min_date = max_date - pd.Timedelta(weeks=Config.EMBED_WINDOW_WEEKS)
        df = transactions[transactions["t_dat"] > min_date].copy()
        print(
            f"Training data size: {len(df)} rows (Window: {Config.EMBED_WINDOW_WEEKS} weeks)"
        )

        # Build Vocabulary
        print("Building vocabulary...")
        article_counts = df["article_id"].value_counts()
        valid_articles = article_counts[
            article_counts >= Config.W2V_MIN_COUNT
        ].index.tolist()

        self.vocab = {str(aid): i for i, aid in enumerate(valid_articles)}
        self.inverse_vocab = {i: str(aid) for aid, i in self.vocab.items()}
        vocab_size = len(self.vocab)
        print(f"Vocabulary size: {vocab_size}")

        # Generate Pairs
        print("Generating training pairs (Sliding Window)...")
        user_histories = df.groupby("customer_id")["article_id"].apply(list)

        centers = []
        contexts = []
        window_size = Config.W2V_WINDOW

        # Iterate over user histories to generate pairs
        for history in user_histories:
            if len(history) < 2:
                continue

            # Convert to indices, skipping OOV items
            indices = [
                self.vocab[str(aid)] for aid in history if str(aid) in self.vocab
            ]
            if len(indices) < 2:
                continue

            for i, target in enumerate(indices):
                start = max(0, i - window_size)
                end = min(len(indices), i + window_size + 1)

                for j in range(start, end):
                    if i != j:
                        centers.append(target)
                        contexts.append(indices[j])

        print(f"Generated {len(centers)} pairs.")

        # Create DataLoader
        centers_tensor = torch.tensor(centers, dtype=torch.long)
        contexts_tensor = torch.tensor(contexts, dtype=torch.long)
        dataset = TensorDataset(centers_tensor, contexts_tensor)

        # Large batch size for speed
        batch_size = 4096
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        # Initialize Model
        model = SkipGramModel(vocab_size, Config.EMBED_DIM).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        # Training Loop
        print(f"Starting training on {self.device}...")
        model.train()

        for epoch in range(Config.W2V_EPOCHS):
            total_loss = 0
            for batch_i, (u_batch, v_batch) in enumerate(dataloader):
                u_batch = u_batch.to(self.device)
                v_batch = v_batch.to(self.device)

                # Sample negatives
                v_neg = torch.randint(
                    0,
                    vocab_size,
                    (u_batch.size(0), Config.W2V_NEGATIVE),
                    device=self.device,
                )

                optimizer.zero_grad()
                loss = model(u_batch, v_batch, v_neg)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{Config.W2V_EPOCHS} - Loss: {avg_loss:.6f}")

        # Extract and Save Embeddings
        # We use the target embeddings (u_embeddings) as the representation
        self.embeddings = model.u_embeddings.weight.data.cpu().numpy()

        print(f"Saving embeddings to {embed_path}")
        with open(vocab_path, "w") as f:
            json.dump(self.vocab, f)
        np.save(embed_path, self.embeddings)

    def get_embedding(self, article_id):
        """
        Returns the embedding vector for a specific article_id.
        Returns None if the article was not in the training vocabulary.
        """
        idx = self.vocab.get(str(article_id))
        if idx is not None:
            return self.embeddings[idx]
        return None

    def get_user_embeddings(self, transactions: pd.DataFrame):
        """
        Computes user embeddings by averaging the embeddings of items in their history.

        Args:
            transactions: DataFrame containing 'customer_id' and 'article_id'.

        Returns:
            dict: Mapping from customer_id to numpy array (embedding vector).
        """
        if self.embeddings is None:
            raise ValueError("Model not trained or loaded. Call fit() first.")

        print("Computing user embeddings...")

        df = transactions.copy()
        df["article_id"] = df["article_id"].astype(str)

        # Map article_id to index
        df["article_idx"] = df["article_id"].map(self.vocab)

        # Drop interactions with OOV items
        df = df.dropna(subset=["article_idx"])
        df["article_idx"] = df["article_idx"].astype(int)

        # Group by customer and aggregate indices
        user_groups = df.groupby("customer_id")["article_idx"].apply(list)

        user_embeddings = {}

        # Compute mean embedding for each user
        # Note: Iterating over groups is acceptable for ~1M users given the vector operations are fast
        for cust_id, indices in user_groups.items():
            if not indices:
                continue

            # Retrieve vectors [n_items, embed_dim]
            vecs = self.embeddings[indices]

            # Compute mean [embed_dim]
            mean_vec = np.mean(vecs, axis=0)
            user_embeddings[cust_id] = mean_vec

        return user_embeddings

    def find_similar_items(self, article_id, k=10):
        """
        Finds the k most similar items to the given article_id using Cosine Similarity.
        """
        if self.embeddings is None:
            return []

        target_vec = self.get_embedding(article_id)
        if target_vec is None:
            return []

        # Normalize embeddings for Cosine Similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normed_embeds = self.embeddings / norms

        target_norm = target_vec / np.linalg.norm(target_vec)

        # Compute dot product
        scores = np.dot(normed_embeds, target_norm)

        # Get top K
        top_indices = np.argsort(scores)[-k:][::-1]

        return [(self.inverse_vocab[i], scores[i]) for i in top_indices]
