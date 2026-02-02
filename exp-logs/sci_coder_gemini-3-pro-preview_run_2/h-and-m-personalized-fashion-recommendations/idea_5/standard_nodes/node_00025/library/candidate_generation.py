import pandas as pd
import numpy as np
import torch
import gc
import os
from pathlib import Path
from tqdm import tqdm
from library import config
from library import utils
from library import data_loader
from library import sequential_encoder
from library import heuristics


class CandidateOrchestrator:
    """
    Orchestrates the retrieval of candidates from multiple sources:
    A. Co-occurrence (Structure)
    B. Sequential Model (Sequence)
    C. Repurchase (Habit)
    D. Global Popularity (Trend)
    """

    def __init__(self):
        self.device = config.DEVICE

    def generate_candidates(
        self, history_df, target_customer_ids, cache_path, load_cached_data=False
    ):
        """
        Generates a combined list of candidates for the target customers.

        Args:
            history_df (pd.DataFrame): The transaction history to learn from.
            target_customer_ids (list/array): The customer_ids to generate candidates for.
            cache_path (Path): Path to save/load the final candidate dataframe.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: DataFrame with columns ['customer_id', 'article_id'] containing unique pairs.
        """
        cache_path = Path(cache_path)

        # 1. Check Cache
        if load_cached_data and cache_path.exists():
            print(f"Loading combined candidates from {cache_path}...")
            return pd.read_parquet(cache_path)

        print("Starting Multi-Source Candidate Retrieval...")

        # Ensure target_customer_ids is a unique list/array
        target_customer_ids = np.unique(target_customer_ids)

        candidates_list = []

        # ==========================================
        # Source A: Co-occurrence (Structure)
        # ==========================================
        print("\n--- Source A: Co-occurrence Retrieval ---")
        cooc_model = heuristics.CooccurrenceMatrix()
        # We pass load_cached_data to fit. If re-running with different data (e.g. val vs test),
        # caller should ensure load_cached_data is False or cache files are managed.
        cooc_model.fit(
            history_df,
            weeks=config.COOC_HISTORY_WEEKS,
            load_cached_data=load_cached_data,
        )

        # Retrieve
        cooc_candidates_dict = cooc_model.get_candidates(
            history_df, target_customer_ids, top_k=config.COOC_TOP_K
        )

        # Convert dict to DataFrame
        cooc_rows = []
        for cid, items in cooc_candidates_dict.items():
            for item in items:
                cooc_rows.append({"customer_id": cid, "article_id": item})

        if cooc_rows:
            df_cooc = pd.DataFrame(cooc_rows)
            candidates_list.append(df_cooc)
            print(f"Source A generated {len(df_cooc)} candidates.")
        else:
            print("Source A generated 0 candidates.")

        del cooc_model, cooc_candidates_dict, cooc_rows
        gc.collect()

        # ==========================================
        # Source B: Sequential Model (Sequence)
        # ==========================================
        print("\n--- Source B: Sequential Deep Retrieval ---")
        # 1. Preprocess
        seq_data = data_loader.preprocess_sequences(
            history_df,
            min_history=config.SEQ_MIN_HISTORY,
            max_seq_len=config.SEQ_CONFIG["max_seq_len"],
            load_cached_data=load_cached_data,
        )

        # 2. Train
        seq_model = sequential_encoder.train_sequential_model(
            seq_data, params=config.SEQ_CONFIG, load_cached_data=load_cached_data
        )

        # 3. Extract Embeddings
        user_embs, item_embs = sequential_encoder.extract_embeddings(
            seq_model, seq_data
        )

        # 4. Perform Retrieval
        # Map target customers to the indices in user_embs
        # seq_data['customer_ids'] contains the customer_id for each row in user_embs
        train_users = seq_data["customer_ids"]  # numpy array of customer_ids

        # Create a map from customer_id -> index in user_embs
        user_id_to_idx = {uid: i for i, uid in enumerate(train_users)}

        # Identify which target customers have embeddings
        valid_targets = []
        valid_target_indices = []

        for uid in target_customer_ids:
            if uid in user_id_to_idx:
                valid_targets.append(uid)
                valid_target_indices.append(user_id_to_idx[uid])

        print(
            f"Sequential Model coverage: {len(valid_targets)}/{len(target_customer_ids)} target users."
        )

        if valid_targets:
            # Convert to numpy for indexing
            valid_target_indices = np.array(valid_target_indices)
            target_user_embs = user_embs[valid_target_indices]  # (N_targets, Embed_Dim)

            # Normalize for Cosine Similarity (optional, but standard for retrieval)
            # SASRec usually uses dot product, but normalization helps stability
            # We stick to dot product as per standard SASRec implementation unless specified otherwise.

            # Batch Retrieval to save memory
            # Matrix size: Batch_Size * Vocab_Size
            batch_size = 1000
            vocab_size = item_embs.shape[0]
            top_k = 20  # Number of sequential candidates

            seq_rows = []

            # Move item embeddings to GPU if possible for fast search
            if torch.cuda.is_available():
                item_embs_tensor = torch.tensor(item_embs, device=config.DEVICE)
            else:
                item_embs_tensor = torch.tensor(item_embs)

            reverse_article_map = seq_data["reverse_article_map"]

            for i in tqdm(
                range(0, len(valid_targets), batch_size), desc="Seq Retrieval"
            ):
                end = min(i + batch_size, len(valid_targets))
                batch_users_emb = target_user_embs[i:end]
                batch_cust_ids = valid_targets[i:end]

                # To Tensor
                if torch.cuda.is_available():
                    batch_users_tensor = torch.tensor(
                        batch_users_emb, device=config.DEVICE
                    )
                else:
                    batch_users_tensor = torch.tensor(batch_users_emb)

                # Dot Product: (Batch, Dim) @ (Vocab, Dim).T -> (Batch, Vocab)
                scores = torch.matmul(batch_users_tensor, item_embs_tensor.t())

                # Top K
                # We ignore index 0 (padding) usually, but it likely has low score if untrained or masked
                top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)

                top_indices = top_indices.cpu().numpy()

                for j, cust_id in enumerate(batch_cust_ids):
                    indices = top_indices[j]
                    for idx in indices:
                        if idx == 0:
                            continue  # Skip padding
                        if idx in reverse_article_map:
                            article_id = reverse_article_map[idx]
                            seq_rows.append(
                                {"customer_id": cust_id, "article_id": article_id}
                            )

            if seq_rows:
                df_seq = pd.DataFrame(seq_rows)
                candidates_list.append(df_seq)
                print(f"Source B generated {len(df_seq)} candidates.")

            # Cleanup GPU memory
            del item_embs_tensor, batch_users_tensor, scores
            torch.cuda.empty_cache()

        del seq_data, seq_model, user_embs, item_embs
        gc.collect()

        # ==========================================
        # Source C: Repurchase (Habit)
        # ==========================================
        print("\n--- Source C: Repurchase Retrieval ---")
        df_repurchase = heuristics.get_repurchase_candidates(
            history_df, customer_ids=target_customer_ids, limit=config.REPURCHASE_LIMIT
        )
        if not df_repurchase.empty:
            candidates_list.append(df_repurchase)
            print(f"Source C generated {len(df_repurchase)} candidates.")

        # ==========================================
        # Source D: Global Popularity (Trend)
        # ==========================================
        print("\n--- Source D: Popularity Retrieval ---")
        pop_items = heuristics.get_global_popularity(
            history_df, days=config.TREND_WINDOW_DAYS, top_k=12
        )

        # We need to broadcast these items to all users
        # To be efficient, we might only add these if a user has < 12 candidates total,
        # but for the "Candidate Generation" phase, we usually just add them to the pool.
        # Creating a dataframe of (N_users * 12) rows is heavy but necessary for the ranker
        # to consider them explicitly.

        # Optimization: Instead of creating a massive dataframe here, we can handle popularity
        # filling in the merging step or just append to the unique list.
        # Let's create a compact representation and expand.

        # For this implementation, we will append them.
        # To avoid memory explosion, we can iterate or use repeat.
        print(f"Top trending items: {pop_items}")

        # Create a DataFrame with all targets and these items
        # Cartesian product
        # Using numpy repeat/tile
        n_targets = len(target_customer_ids)
        n_items = len(pop_items)

        pop_cust_col = np.repeat(target_customer_ids, n_items)
        pop_item_col = np.tile(pop_items, n_targets)

        df_pop = pd.DataFrame({"customer_id": pop_cust_col, "article_id": pop_item_col})
        candidates_list.append(df_pop)
        print(f"Source D generated {len(df_pop)} candidates.")

        # ==========================================
        # Merge & Deduplicate
        # ==========================================
        print("\nMerging candidates...")
        if not candidates_list:
            print("Warning: No candidates generated!")
            return pd.DataFrame(columns=["customer_id", "article_id"])

        full_candidates = pd.concat(candidates_list, ignore_index=True)

        print(f"Total raw candidates: {len(full_candidates)}")

        # Deduplicate
        full_candidates = full_candidates.drop_duplicates(
            subset=["customer_id", "article_id"]
        )
        print(f"Unique candidates: {len(full_candidates)}")

        # Save to cache
        print(f"Saving candidates to {cache_path}...")
        # Ensure parent dir exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        full_candidates.to_parquet(cache_path, index=False)

        return full_candidates
