import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from library.config import WORKING_DIR, NUM_NEIGHBORS, TOP_K_INSTANCES, RANDOM_STATE
from library.utils import seed_everything


class NeighborhoodExtractor:
    """
    Extracts Decoupled Multi-Resolution Neighborhood features for the Stage 2 model.
    Computes features based on the similarity of markdown cells to code cells within the same notebook
    using two views: Lexical (TF-IDF) and Latent (SVD).
    """

    def __init__(self, num_neighbors=NUM_NEIGHBORS, top_k=TOP_K_INSTANCES):
        self.num_neighbors = num_neighbors
        self.top_k = top_k
        seed_everything(RANDOM_STATE)

    def extract_neighborhood_features(
        self, df_cells, vectorizer, split="train", load_cached_data=True
    ):
        """
        Main method to generate neighborhood features. Handles caching.

        Args:
            df_cells (pd.DataFrame): DataFrame containing all cells (code and markdown).
            vectorizer (TextVectorizer): Fitted vectorizer instance.
            split (str): 'train', 'val', or 'test'. Used for cache naming.
            load_cached_data (bool): Whether to load from disk if available.

        Returns:
            pd.DataFrame: DataFrame containing feature columns for markdown cells.
        """
        os.makedirs(WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(WORKING_DIR, f"{split}_neighborhood_features.parquet")

        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading cached neighborhood features from {cache_path}...")
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print(f"Computing neighborhood features for {split} set...")

        # 1. Vectorize all text
        # We need vectors for both code and markdown to compute similarity
        print("Vectorizing text data...")
        tfidf_mat, svd_mat = vectorizer.transform(df_cells["source"])

        # 2. Prepare output storage
        # We only generate features for markdown cells
        md_mask = df_cells["cell_type"] == "markdown"
        df_md = df_cells[md_mask].copy().reset_index(drop=True)

        # Create a mapping from cell_id to matrix index
        cell_id_to_idx = {cid: i for i, cid in enumerate(df_cells["cell_id"])}

        # Group cells by notebook for processing
        nb_groups = df_cells.groupby("notebook_id")

        features_list = []

        # Iterate over unique notebooks present in the markdown set
        unique_nbs = df_md["notebook_id"].unique()

        # Pre-define feature keys to ensure consistency
        cols = ["cell_id", "notebook_id"]
        for view in ["lex", "lat"]:
            for k in range(1, self.top_k + 1):
                cols.append(f"{view}_top{k}_sim")
                cols.append(f"{view}_top{k}_rank")
            cols.append(f"{view}_mean_rank")
            cols.append(f"{view}_std_rank")

        # Iterate over notebooks
        for nb_id in unique_nbs:
            nb_cells = nb_groups.get_group(nb_id)

            # Separate Code and MD
            code_cells = nb_cells[nb_cells["cell_type"] == "code"]
            md_cells = nb_cells[nb_cells["cell_type"] == "markdown"]

            if len(md_cells) == 0:
                continue

            # If no code cells, fill with defaults
            if len(code_cells) == 0:
                for _, row in md_cells.iterrows():
                    feat = {c: 0.0 for c in cols if c not in ["cell_id", "notebook_id"]}
                    feat["cell_id"] = row["cell_id"]
                    feat["notebook_id"] = nb_id
                    # Set ranks to 0.5 (neutral)
                    for view in ["lex", "lat"]:
                        for k in range(1, self.top_k + 1):
                            feat[f"{view}_top{k}_rank"] = 0.5
                        feat[f"{view}_mean_rank"] = 0.5
                    features_list.append(feat)
                continue

            # Get indices for vectors
            code_indices = [cell_id_to_idx[cid] for cid in code_cells["cell_id"]]
            md_indices = [cell_id_to_idx[cid] for cid in md_cells["cell_id"]]

            # Extract vectors
            # Lexical (Sparse)
            code_vecs_lex = tfidf_mat[code_indices]
            md_vecs_lex = tfidf_mat[md_indices]

            # Latent (Dense)
            code_vecs_lat = svd_mat[code_indices]
            md_vecs_lat = svd_mat[md_indices]

            # Calculate Code Ranks (Anchors)
            # We assume code cells are in correct order in the input df_cells
            n_code = len(code_cells)
            if n_code > 1:
                code_ranks = np.arange(n_code) / (n_code - 1)
            else:
                code_ranks = np.array([0.0])

            # --- Compute Similarities ---

            # 1. Lexical Similarity (Sparse Dot Product)
            # Result: (n_md, n_code)
            sim_lex = md_vecs_lex.dot(code_vecs_lex.T).toarray()

            # 2. Latent Similarity (Dense Dot Product)
            # Result: (n_md, n_code)
            sim_lat = np.dot(md_vecs_lat, code_vecs_lat.T)

            # --- Extract Features per MD cell ---
            md_cell_ids = md_cells["cell_id"].values

            for i, cell_id in enumerate(md_cell_ids):
                feat = {"cell_id": cell_id, "notebook_id": nb_id}

                # Process both views
                for view, sim_matrix in [("lex", sim_lex), ("lat", sim_lat)]:
                    sims = sim_matrix[i]

                    # Sort indices by similarity descending
                    sorted_indices = np.argsort(sims)

                    # Top-K Instances
                    k_actual = min(self.top_k, n_code)
                    top_k_idx = sorted_indices[-k_actual:][::-1]

                    for k in range(1, self.top_k + 1):
                        if k <= k_actual:
                            idx = top_k_idx[k - 1]
                            feat[f"{view}_top{k}_sim"] = float(sims[idx])
                            feat[f"{view}_top{k}_rank"] = float(code_ranks[idx])
                        else:
                            feat[f"{view}_top{k}_sim"] = 0.0
                            feat[f"{view}_top{k}_rank"] = 0.5

                    # Smoothed Neighborhood
                    n_smooth = min(self.num_neighbors, n_code)
                    smooth_idx = sorted_indices[-n_smooth:]

                    neighbor_ranks = code_ranks[smooth_idx]

                    feat[f"{view}_mean_rank"] = float(np.mean(neighbor_ranks))
                    feat[f"{view}_std_rank"] = (
                        float(np.std(neighbor_ranks)) if n_smooth > 1 else 0.0
                    )

                features_list.append(feat)

        # Create DataFrame
        df_features = pd.DataFrame(features_list)

        # Save to cache
        print(f"Saving neighborhood features to {cache_path}...")
        df_features.to_parquet(cache_path, index=False)

        return df_features


def assemble_stage2_features(df_neighborhood, df_ridge_preds, df_meta):
    """
    Combines neighborhood features, Ridge predictions, and metadata to form the
    final Stage 2 dataset. Computes Cross-View Interaction features.

    Args:
        df_neighborhood (pd.DataFrame): Output from extract_neighborhood_features.
        df_ridge_preds (pd.DataFrame): DataFrame with 'cell_id' and 'pred' (Ridge prediction).
        df_meta (pd.DataFrame): Metadata/Dataframe containing 'notebook_id' and 'cell_type'.
                                Used to derive notebook-level stats.

    Returns:
        pd.DataFrame: Ready-to-train DataFrame.
    """
    # 1. Merge Neighborhood Features with Ridge Predictions
    # Ensure we only keep cells present in both (markdown cells)
    df = pd.merge(df_neighborhood, df_ridge_preds, on="cell_id", how="inner")

    # Rename ridge prediction column for clarity
    if "pred" in df.columns:
        df = df.rename(columns={"pred": "ridge_rank"})

    # 2. Compute Cross-View Interaction Features (Deltas)
    # Ridge vs Lexical Top 1
    df["delta_ridge_lex"] = (df["ridge_rank"] - df["lex_top1_rank"]).abs()

    # Ridge vs Latent Top 1
    df["delta_ridge_lat"] = (df["ridge_rank"] - df["lat_top1_rank"]).abs()

    # Lexical vs Latent (Disagreement)
    df["delta_lex_lat"] = (df["lex_top1_rank"] - df["lat_top1_rank"]).abs()

    # Ridge vs Lexical Smoothed
    df["delta_ridge_lex_mean"] = (df["ridge_rank"] - df["lex_mean_rank"]).abs()

    # Ridge vs Latent Smoothed
    df["delta_ridge_lat_mean"] = (df["ridge_rank"] - df["lat_mean_rank"]).abs()

    # 3. Add Notebook Metadata Features (Length, Ratio)
    if "cell_type" in df_meta.columns:
        # Calculate notebook stats
        nb_stats = (
            df_meta.groupby("notebook_id")
            .apply(
                lambda x: pd.Series(
                    {
                        "n_total": len(x),
                        "n_code": (x["cell_type"] == "code").sum(),
                        "n_md": (x["cell_type"] == "markdown").sum(),
                    }
                )
            )
            .reset_index()
        )

        nb_stats["md_ratio"] = nb_stats["n_md"] / nb_stats["n_total"]

        # Merge stats
        df = pd.merge(
            df,
            nb_stats[["notebook_id", "n_total", "md_ratio"]],
            on="notebook_id",
            how="left",
        )

    return df
