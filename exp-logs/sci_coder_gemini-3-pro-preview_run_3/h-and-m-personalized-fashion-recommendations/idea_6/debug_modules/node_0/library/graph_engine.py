import os
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from datetime import timedelta
from tqdm import tqdm

from library.config import Config
from library.utils import Timer, print_memory_usage, normalize_matrix


class GraphEngine:
    """
    Constructs sparse graphs for the Multi-Temporal Retrieval stage.
    1. Short-Term Transition Matrix (Trends)
    2. Long-Term Transition Matrix (Style)
    3. Visual Similarity Graph (Content)
    4. User History Matrix (User Representation)
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.working_dir = Config.WORKING_DIR
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def build_graphs(
        self,
        train_df: pd.DataFrame,
        embeddings: np.ndarray,
        load_cached_data: bool = True,
    ):
        """
        Orchestrates the creation of all graph structures.
        """
        # Check cache existence
        cache_files = [
            Config.CACHE_GRAPH_SHORT,
            Config.CACHE_GRAPH_LONG,
            Config.CACHE_GRAPH_VISUAL,
            Config.CACHE_USER_HISTORY,
        ]

        if load_cached_data and all(f.exists() for f in cache_files):
            print("All graph structures found in cache. Skipping generation.")
            return

        print("Generating sparse graphs...")

        # Ensure dates are datetime
        if not np.issubdtype(train_df["t_dat"].dtype, np.datetime64):
            train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])

        max_date = train_df["t_dat"].max()
        print(f"Max Date in training data: {max_date}")

        # 1. Build User History (Full with Decay)
        # This is used as the 'User Vector' U during retrieval
        self._build_user_history(train_df, max_date)
        gc.collect()

        # 2. Build Short-Term Graph (Last 28 Days)
        # Captures immediate trends
        start_short = max_date - timedelta(days=Config.SHORT_TERM_WINDOW)
        print(f"Building Short-Term Graph (Start: {start_short})...")
        self._build_transition_matrix(
            train_df,
            start_date=start_short,
            end_date=max_date,
            output_path=Config.CACHE_GRAPH_SHORT,
            top_k=Config.RETRIEVAL_TOP_K,
        )
        gc.collect()

        # 3. Build Long-Term Graph (Days 28 to 140)
        # Captures long-term style/seasonality
        end_long = start_short
        start_long = max_date - timedelta(days=Config.LONG_TERM_WINDOW)
        print(f"Building Long-Term Graph ({start_long} to {end_long})...")
        self._build_transition_matrix(
            train_df,
            start_date=start_long,
            end_date=end_long,
            output_path=Config.CACHE_GRAPH_LONG,
            top_k=Config.RETRIEVAL_TOP_K,
        )
        gc.collect()

        # 4. Build Visual Graph
        # Captures visual similarity via Embeddings
        print("Building Visual Similarity Graph...")
        self._build_visual_graph(
            embeddings, k=Config.VISUAL_KNN_K, output_path=Config.CACHE_GRAPH_VISUAL
        )
        gc.collect()

        print_memory_usage("Graph Generation Complete")

    def _build_user_history(self, df: pd.DataFrame, ref_date):
        """
        Constructs the User-Item interaction matrix with time decay.
        Saves to Config.CACHE_USER_HISTORY.
        """
        with Timer("Build User History"):
            # Calculate days elapsed
            # We use a simple approximation: (ref_date - t_dat).days
            # Vectorized calculation
            delta_days = (ref_date - df["t_dat"]).dt.days.values.astype(np.float32)

            # Apply Power-Law Decay: w = 1 / (1 + days)
            # This emphasizes recent interactions in the user vector
            weights = 1.0 / (1.0 + delta_days)

            # Construct Sparse Matrix (Users x Articles)
            # Duplicate entries (multiple purchases) are summed by default in coo_matrix -> csr conversion
            row = df["customer_id"].values
            col = df["article_id"].values

            # Dimensions
            n_users = int(row.max()) + 1
            n_items = int(col.max()) + 1
            # Ensure dimensions match the global maps (use max from data or maps)
            # We use the max index observed + padding if necessary, but here we trust the mapping
            # from DataLoader which maps to range [0, N-1].

            user_history = sp.coo_matrix(
                (weights, (row, col)), shape=(n_users, n_items)
            ).tocsr()

            # Normalize rows (User vectors sum to 1 or unit length? L1 is good for 'probability' distribution)
            user_history = normalize_matrix(user_history, axis=1, norm="l1")

            print(f"User History Shape: {user_history.shape}, NNZ: {user_history.nnz}")
            sp.save_npz(Config.CACHE_USER_HISTORY, user_history)

    def _build_transition_matrix(
        self, df: pd.DataFrame, start_date, end_date, output_path, top_k=100
    ):
        """
        Constructs an Item-Item co-visitation matrix (R.T @ R).
        """
        with Timer(f"Transition Matrix"):
            # Filter Data
            mask = (df["t_dat"] > start_date) & (df["t_dat"] <= end_date)
            subset = df.loc[mask]

            if subset.empty:
                print("Warning: No data in specified time range. Saving empty matrix.")
                # Save empty matrix of correct size
                # Need to know total items. We can infer from article map size if available,
                # or just use max id from full df.
                # For safety, we load the map size.
                article_map = np.load(Config.CACHE_ARTICLE_MAP, allow_pickle=True)
                n_items = len(article_map)
                empty = sp.csr_matrix((n_items, n_items))
                sp.save_npz(output_path, empty)
                return

            # Construct User-Item Matrix R for this window
            # We use binary weights or simple count for transition matrix construction
            # to capture "bought together" strength.
            # Time decay within the window is less critical if the window is short,
            # but we can apply weak decay.

            row = subset["customer_id"].values
            col = subset["article_id"].values
            data = np.ones_like(row, dtype=np.float32)

            n_users = subset["customer_id"].max() + 1
            article_map = np.load(Config.CACHE_ARTICLE_MAP, allow_pickle=True)
            n_items = len(article_map)

            R = sp.coo_matrix((data, (row, col)), shape=(n_users, n_items)).tocsr()

            # Binarize R? If a user bought item X 5 times, is the link stronger?
            # Yes, but usually we care about existence. Let's keep counts for strength.

            # Compute Co-occurrence: C = R.T @ R
            # This results in (n_items x n_items)
            print(f"Computing R.T @ R for {len(subset)} transactions...")
            C = R.T.dot(R)

            # Zero out diagonal (Item i -> Item i is not a useful recommendation)
            C.setdiag(0)
            C.eliminate_zeros()

            # Prune to Top-K per row to maintain sparsity
            print(f"Pruning to Top-{top_k} neighbors...")
            C = self._prune_matrix(C, top_k)

            # Normalize rows to create probability distribution P(j|i)
            C = normalize_matrix(C, axis=1, norm="l1")

            print(f"Graph Shape: {C.shape}, NNZ: {C.nnz}")
            sp.save_npz(output_path, C)

    def _build_visual_graph(self, embeddings: np.ndarray, k: int, output_path):
        """
        Constructs a KNN graph using Cosine Similarity on embeddings.
        Uses GPU chunking to handle memory constraints.
        """
        with Timer("Visual KNN Graph"):
            n_items, dim = embeddings.shape

            # Normalize embeddings for Cosine Similarity (L2 norm)
            # (Dot product of normalized vectors == Cosine Similarity)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10  # Avoid div by zero
            embeddings_norm = embeddings / norms

            # Convert to Torch
            # We use float16 for memory efficiency on GPU if supported, else float32
            tensor_emb = torch.from_numpy(embeddings_norm).to(
                self.device, dtype=torch.float32
            )

            # Result lists
            rows_list = []
            cols_list = []
            data_list = []

            # Chunk size
            chunk_size = 5000

            # Process in chunks
            for start_idx in tqdm(range(0, n_items, chunk_size), desc="Visual KNN"):
                end_idx = min(start_idx + chunk_size, n_items)

                # Query Chunk: (Chunk, Dim)
                query = tensor_emb[start_idx:end_idx]

                # Compute Similarity: (Chunk, Dim) @ (Dim, N) -> (Chunk, N)
                sim_matrix = torch.mm(query, tensor_emb.t())

                # Top-K
                # We want top K+1 because the item itself will be index 0 (sim=1.0)
                # We will remove self-loops later
                top_vals, top_inds = torch.topk(sim_matrix, k=k + 1, dim=1)

                # Move to CPU
                top_vals = top_vals.cpu().numpy()
                top_inds = top_inds.cpu().numpy()

                # Build coordinate lists
                # query[i] corresponds to global index start_idx + i
                n_chunk = end_idx - start_idx

                # Create row indices: [start, start, ..., start+1, ...]
                row_indices = np.arange(start_idx, end_idx).repeat(k + 1)
                col_indices = top_inds.flatten()
                values = top_vals.flatten()

                # Filter self-loops (where row_index == col_index)
                mask = row_indices != col_indices

                rows_list.append(row_indices[mask])
                cols_list.append(col_indices[mask])
                data_list.append(values[mask])

                # Memory cleanup
                del sim_matrix, top_vals, top_inds

            # Concatenate
            all_rows = np.concatenate(rows_list)
            all_cols = np.concatenate(cols_list)
            all_data = np.concatenate(data_list)

            # Construct CSR
            visual_graph = sp.coo_matrix(
                (all_data, (all_rows, all_cols)), shape=(n_items, n_items)
            ).tocsr()

            # Normalize
            visual_graph = normalize_matrix(visual_graph, axis=1, norm="l1")

            print(f"Visual Graph Shape: {visual_graph.shape}, NNZ: {visual_graph.nnz}")
            sp.save_npz(output_path, visual_graph)

            # Cleanup GPU
            del tensor_emb
            torch.cuda.empty_cache()

    def _prune_matrix(self, matrix: sp.csr_matrix, k: int) -> sp.csr_matrix:
        """
        Keeps only the top K values per row in a CSR matrix.
        """
        # If matrix is already sparse enough, skip (heuristic: avg degree < k)
        if matrix.nnz / matrix.shape[0] <= k:
            return matrix

        # Iterate over rows and prune
        # This implementation modifies the data arrays in place or reconstructs efficiently

        new_data = []
        new_indices = []
        new_indptr = [0]

        # Access underlying arrays
        mat_data = matrix.data
        mat_indices = matrix.indices
        mat_indptr = matrix.indptr

        for i in range(matrix.shape[0]):
            start = mat_indptr[i]
            end = mat_indptr[i + 1]

            if start == end:
                new_indptr.append(new_indptr[-1])
                continue

            row_data = mat_data[start:end]
            row_inds = mat_indices[start:end]

            if len(row_data) > k:
                # Get indices of top k elements
                # argpartition is faster than sort for finding top k
                top_k_idx = np.argpartition(row_data, -k)[-k:]

                new_data.append(row_data[top_k_idx])
                new_indices.append(row_inds[top_k_idx])
                new_indptr.append(new_indptr[-1] + k)
            else:
                new_data.append(row_data)
                new_indices.append(row_inds)
                new_indptr.append(new_indptr[-1] + len(row_data))

        # Flatten lists
        if new_data:
            new_data = np.concatenate(new_data)
            new_indices = np.concatenate(new_indices)
        else:
            new_data = np.array([])
            new_indices = np.array([])

        return sp.csr_matrix((new_data, new_indices, new_indptr), shape=matrix.shape)
