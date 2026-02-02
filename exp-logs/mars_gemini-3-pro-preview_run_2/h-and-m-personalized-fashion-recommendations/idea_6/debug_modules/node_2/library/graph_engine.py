import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import LGConv
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional

from library.config import GCN_PARAMS, SEED
from library.utils import setup_logger

# Setup logger
logger = setup_logger("graph_engine")


class LightGCN(nn.Module):
    """
    LightGCN model implementation using torch_geometric.
    """

    def __init__(
        self, num_users: int, num_items: int, embedding_dim: int = 64, n_layers: int = 3
    ):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        # Initialize embeddings
        # We use a single embedding table for all nodes (users + items)
        # Users: 0 to num_users - 1
        # Items: num_users to num_users + num_items - 1
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)

        # Initialize weights using Normal distribution (standard for LightGCN)
        nn.init.normal_(self.embedding.weight, std=0.1)

        # Graph Convolution Layer
        # LightGCN uses the same propagation rule at each layer, so we can reuse the class
        self.conv = LGConv()

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Propagates embeddings through the graph.

        Args:
            edge_index: Graph connectivity (2, num_edges).

        Returns:
            Final embeddings for all nodes (users + items).
        """
        # Initial embeddings (Layer 0)
        x = self.embedding.weight

        # Layer combination: LightGCN typically uses a weighted sum of all layers
        # Standard implementation: 1/(K+1) * sum(E_k)
        out = x

        for _ in range(self.n_layers):
            x = self.conv(x, edge_index)
            out = out + x

        return out / (self.n_layers + 1)

    def get_batch_embeddings(
        self,
        edge_index: torch.Tensor,
        user_indices: torch.Tensor,
        pos_item_indices: torch.Tensor,
        neg_item_indices: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Retrieves embeddings for a specific batch of users and items for BPR loss calculation.
        Also returns regularization terms.
        """
        # 1. Get full graph embeddings
        final_embeddings = self.forward(edge_index)

        # 2. Extract specific embeddings for the batch
        # Note: Item indices in the embedding table are shifted by num_users
        u_final = final_embeddings[user_indices]
        pos_i_final = final_embeddings[self.num_users + pos_item_indices]
        neg_i_final = final_embeddings[self.num_users + neg_item_indices]

        # 3. Get initial embeddings for regularization (L2 reg is applied to layer 0)
        u_ego = self.embedding(user_indices)
        pos_i_ego = self.embedding(self.num_users + pos_item_indices)
        neg_i_ego = self.embedding(self.num_users + neg_item_indices)

        return u_final, pos_i_final, neg_i_final, u_ego, pos_i_ego, neg_i_ego


def bpr_loss(
    users_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    user_ego: torch.Tensor,
    pos_ego: torch.Tensor,
    neg_ego: torch.Tensor,
    decay: float,
) -> torch.Tensor:
    """
    Bayesian Personalized Ranking Loss.
    L = -mean(ln(sigmoid(pos_score - neg_score))) + lambda * ||params||^2
    """
    # Calculate scores (dot product)
    pos_scores = torch.sum(users_emb * pos_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_emb, dim=1)

    # BPR Loss
    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    # L2 Regularization
    reg_loss = (
        user_ego.norm(2).pow(2) + pos_ego.norm(2).pow(2) + neg_ego.norm(2).pow(2)
    ) / 2.0

    # Normalize reg loss by batch size usually, or just scale by decay
    # Common implementation scales by batch mean
    reg_loss = (reg_loss / users_emb.shape[0]) * decay

    return loss + reg_loss


def train_graph_embeddings(
    edge_index: torch.Tensor, num_users: int, num_items: int, params: dict = GCN_PARAMS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains the LightGCN model and returns the learned user and item embeddings.

    Args:
        edge_index: Bipartite edge index (2, num_edges). Row 0: User, Row 1: Item.
        num_users: Total number of users.
        num_items: Total number of items.
        params: Hyperparameters dictionary.

    Returns:
        user_embeddings: Numpy array (num_users, dim).
        item_embeddings: Numpy array (num_items, dim).
    """
    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(params["device"])
    logger.info(f"Training LightGCN on {device}...")

    # 1. Prepare Graph Data
    # Shift item indices to create a unified node set
    # User nodes: 0 to num_users-1
    # Item nodes: num_users to num_users+num_items-1

    # Create a copy to avoid modifying original tensor
    train_edge_index = edge_index.clone()
    train_edge_index[1] += num_users

    # Make graph undirected (add reverse edges) for message passing
    # (u -> i) and (i -> u)
    train_edge_index = torch.cat([train_edge_index, train_edge_index.flip(0)], dim=1)
    train_edge_index = train_edge_index.to(device)

    # 2. Initialize Model
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=params["embedding_dim"],
        n_layers=params["n_layers"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # 3. Training Loop
    # We iterate over the original positive edges (before adding reverse edges)
    # The original edge_index has shape [2, E], row 0 users, row 1 items (0-indexed)
    pos_edges = edge_index.t()  # Shape (E, 2)
    num_edges = pos_edges.shape[0]
    batch_size = params["batch_size"]

    model.train()

    for epoch in range(params["epochs"]):
        # Shuffle edges
        perm = torch.randperm(num_edges)
        pos_edges = pos_edges[perm]

        total_loss = 0
        steps = 0

        # Batch processing
        pbar = tqdm(
            range(0, num_edges, batch_size),
            desc=f"Epoch {epoch+1}/{params['epochs']}",
            leave=False,
        )
        for i in pbar:
            # Get batch of positive interactions
            batch_pos_edges = pos_edges[i : i + batch_size]
            users = batch_pos_edges[:, 0].to(device)
            pos_items = batch_pos_edges[:, 1].to(device)

            # Negative Sampling
            # Sample N negatives for each positive
            # For simplicity and speed in this competition context, we sample 1 negative per positive
            # or expand if params['neg_samples'] > 1.
            # Standard BPR usually takes 1 neg per pos, but we can do more.
            # Let's stick to 1-to-1 for memory efficiency on the GPU unless specified otherwise.
            # If params['neg_samples'] > 1, we just repeat the users and pos_items.

            n_neg = params.get("neg_samples", 1)

            # Expand for negative sampling
            # users: [u1, u2] -> [u1, u1, u2, u2]
            users_expanded = users.repeat_interleave(n_neg)
            pos_items_expanded = pos_items.repeat_interleave(n_neg)

            # Sample negatives
            # Randomly select items
            neg_items = torch.randint(
                0, num_items, (users_expanded.size(0),), device=device
            )

            # Note: We skip strict checking (if neg is actually pos) for speed.
            # In sparse datasets like H&M, collision probability is extremely low.

            optimizer.zero_grad()

            # Forward pass
            u_emb, pos_emb, neg_emb, u_ego, pos_ego, neg_ego = (
                model.get_batch_embeddings(
                    train_edge_index, users_expanded, pos_items_expanded, neg_items
                )
            )

            # Loss calculation
            loss = bpr_loss(
                u_emb, pos_emb, neg_emb, u_ego, pos_ego, neg_ego, decay=params["decay"]
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if steps % 100 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / steps
        logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}")

    # 4. Extract Embeddings
    logger.info("Extracting final embeddings...")
    model.eval()
    with torch.no_grad():
        final_embeddings = model.forward(train_edge_index)

        # Split back into users and items
        user_embeddings = final_embeddings[:num_users].cpu().numpy()
        item_embeddings = final_embeddings[num_users:].cpu().numpy()

    logger.info(
        f"Training complete. User Emb: {user_embeddings.shape}, Item Emb: {item_embeddings.shape}"
    )

    return user_embeddings, item_embeddings


def get_embeddings(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    user_indices: Optional[np.ndarray] = None,
    item_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper to retrieve specific embeddings from the full matrices.
    If indices are None, returns the full matrices.
    """
    u_emb = user_embeddings if user_indices is None else user_embeddings[user_indices]
    i_emb = item_embeddings if item_indices is None else item_embeddings[item_indices]
    return u_emb, i_emb
