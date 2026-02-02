import pandas as pd
import numpy as np
import os
from pathlib import Path
from library.config import (
    WORKING_DIR,
    CACHE_RANKER_DATASET,
    CACHE_CANDIDATES_TRAIN,
    CACHE_CANDIDATES_TEST,
    ARTICLES_PATH,
    CUSTOMERS_PATH,
    USER_FEATURES,
    ITEM_FEATURES,
    SEED,
    USER_ID_COL,
    ITEM_ID_COL,
    TARGET_COL,
    TOP_K_RETRIEVAL,
    N_CPUS,
)
from library.retrieval import CooccurrenceRecommender
from library.visual_encoder import extract_all_embeddings

# Set seeds
np.random.seed(SEED)


class RankerDatasetBuilder:
    """
    Constructs the dataset for Stage 2 (Ranking) by merging retrieval candidates
    with metadata and computing interaction features (e.g., visual similarity).
    """

    def __init__(self):
        self.articles = None
        self.customers = None
        self.embeddings = None
        self.user_profiles = None

    def _generate_image_path(self, article_ids):
        """
        Vectorized generation of image paths from article_ids.
        """
        # Ensure string and padding
        ids = article_ids.astype(str).str.zfill(10)
        return "images/" + ids.str.slice(0, 3) + "/" + ids + ".jpg"

    def _load_and_prep_metadata(self):
        """
        Loads and preprocesses articles and customers metadata.
        """
        print("Loading metadata for feature engineering...")

        # 1. Articles
        self.articles = pd.read_csv(ARTICLES_PATH, dtype={ITEM_ID_COL: str})
        self.articles[ITEM_ID_COL] = self.articles[ITEM_ID_COL].str.zfill(10)

        # Encode categorical features in articles
        # We assume _no columns are already numerical representations suitable for LGBM
        # We just need to ensure they are int
        for col in ITEM_FEATURES:
            if col in self.articles.columns:
                self.articles[col] = (
                    pd.to_numeric(self.articles[col], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                )

        # 2. Customers
        self.customers = pd.read_csv(CUSTOMERS_PATH, dtype={USER_ID_COL: str})

        # Encode categorical features in customers
        # club_member_status
        if "club_member_status" in self.customers.columns:
            self.customers["club_member_status"] = (
                self.customers["club_member_status"].astype("category").cat.codes
            )

        # fashion_news_frequency
        if "fashion_news_frequency" in self.customers.columns:
            self.customers["fashion_news_frequency"] = (
                self.customers["fashion_news_frequency"].astype("category").cat.codes
            )

        # Fill NaNs in age
        if "age" in self.customers.columns:
            self.customers["age"] = self.customers["age"].fillna(
                self.customers["age"].mean()
            )

    def _get_embeddings(self):
        """
        Loads or computes image embeddings for all articles.
        """
        if self.embeddings is not None:
            return self.embeddings

        # Generate paths for all articles
        paths = self._generate_image_path(self.articles[ITEM_ID_COL])

        # Extract
        print("Retrieving image embeddings...")
        raw_embeddings = extract_all_embeddings(
            article_ids=self.articles[ITEM_ID_COL].values,
            image_paths=paths.values,
            load_cached_data=True,
        )

        # Normalize embeddings for Cosine Similarity
        # L2 norm
        print("Normalizing embeddings...")
        normalized_embeddings = {}
        for aid, vec in raw_embeddings.items():
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                normalized_embeddings[aid] = vec / norm
            else:
                normalized_embeddings[aid] = vec  # Keep zero vector if zero

        self.embeddings = normalized_embeddings
        return self.embeddings

    def _compute_user_profiles(self, history_df):
        """
        Computes user visual profiles based on history.
        Profile = Mean of normalized embeddings of purchased items.
        """
        print("Computing user visual profiles...")
        embeddings = self._get_embeddings()

        # Filter history for items we have embeddings for
        valid_history = history_df[
            history_df[ITEM_ID_COL].isin(embeddings.keys())
        ].copy()

        if valid_history.empty:
            return {}

        # Map article_id to embedding
        # Doing this efficiently:
        # 1. Get unique articles in history
        unique_arts = valid_history[ITEM_ID_COL].unique()
        # 2. Create a matrix of embeddings for these articles
        art_to_idx = {art: i for i, art in enumerate(unique_arts)}
        emb_dim = len(next(iter(embeddings.values())))
        emb_matrix = np.zeros((len(unique_arts), emb_dim), dtype=np.float32)

        for art, idx in art_to_idx.items():
            emb_matrix[idx] = embeddings[art]

        # 3. Map history to indices
        valid_history["emb_idx"] = valid_history[ITEM_ID_COL].map(art_to_idx)

        # 4. Group by user and sum embeddings
        # Optimization: Sum embeddings per user
        # user_id -> list of emb_indices
        user_groups = valid_history.groupby(USER_ID_COL)["emb_idx"].apply(list)

        profiles = {}
        for user, indices in user_groups.items():
            # Gather vectors
            vecs = emb_matrix[indices]
            # Mean
            mean_vec = np.mean(vecs, axis=0)
            # Normalize profile
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-6:
                profiles[user] = mean_vec / norm
            else:
                profiles[user] = mean_vec

        self.user_profiles = profiles
        return profiles

    def _compute_visual_similarity(self, candidates_df):
        """
        Computes cosine similarity between user profile and candidate item.
        """
        print("Computing visual similarity features...")
        if self.user_profiles is None:
            # Should have been computed via _compute_user_profiles
            return np.zeros(len(candidates_df))

        embeddings = self._get_embeddings()

        # We need to align with candidates_df rows
        user_ids = candidates_df[USER_ID_COL].values
        item_ids = candidates_df[ITEM_ID_COL].values

        n_samples = len(candidates_df)
        sim_scores = np.zeros(n_samples, dtype=np.float32)

        # Define a zero vector
        emb_dim = 512  # ResNet18 default
        if embeddings:
            emb_dim = len(next(iter(embeddings.values())))
        zero_vec = np.zeros(emb_dim, dtype=np.float32)

        # Faster approach:
        # 1. Unique users in candidates, Unique items in candidates.
        # 2. Create matrices.
        # 3. Index into matrices.

        # Unique mapping
        u_unique = np.unique(user_ids)
        i_unique = np.unique(item_ids)

        u_map = {u: i for i, u in enumerate(u_unique)}
        i_map = {item: i for i, item in enumerate(i_unique)}

        # Build matrices
        u_mat = np.zeros((len(u_unique), emb_dim), dtype=np.float32)
        for u, idx in u_map.items():
            u_mat[idx] = self.user_profiles.get(u, zero_vec)

        i_mat = np.zeros((len(i_unique), emb_dim), dtype=np.float32)
        for item, idx in i_map.items():
            i_mat[idx] = embeddings.get(item, zero_vec)

        # Map original arrays to indices
        u_indices = np.array([u_map[u] for u in user_ids])
        i_indices = np.array([i_map[i] for i in item_ids])

        # Compute dot product (cosine sim since normalized)
        # Process in chunks to save memory
        chunk_size = 1000000
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            u_batch = u_mat[u_indices[start:end]]
            i_batch = i_mat[i_indices[start:end]]
            sim_scores[start:end] = np.sum(u_batch * i_batch, axis=1)

        return sim_scores

    def _compute_item_popularity(self, history_df, candidates_df):
        """
        Computes item popularity from history and merges into candidates.
        Popularity is defined as log(count + 1) of purchases in the history window.
        """
        print("Computing item popularity features...")
        # Calculate raw counts
        pop_counts = history_df[ITEM_ID_COL].value_counts().reset_index()
        pop_counts.columns = [ITEM_ID_COL, "raw_pop"]

        # Log transform
        pop_counts["item_popularity"] = np.log1p(pop_counts["raw_pop"])

        # Merge
        # We use left join on candidates to keep order
        merged = candidates_df.merge(
            pop_counts[[ITEM_ID_COL, "item_popularity"]], on=ITEM_ID_COL, how="left"
        )

        # Fill NaNs (items not in history) with 0
        merged["item_popularity"] = merged["item_popularity"].fillna(0.0)

        return merged["item_popularity"]

    def build_train_set(self, train_df, val_df, load_cached_data=True):
        """
        Builds the labeled training dataset.

        1. Fit Retrieval on train_df.
        2. Generate candidates for users in val_df.
        3. Label candidates (1 if in val_df, 0 otherwise).
        4. Compute features.
        """
        # Cache check
        if load_cached_data and CACHE_RANKER_DATASET.exists():
            print(f"Loading cached ranker dataset from {CACHE_RANKER_DATASET}...")
            return pd.read_parquet(CACHE_RANKER_DATASET)

        print("Building Ranker Train Set from scratch...")

        # 1. Retrieval
        # Fit recommender on training history
        recommender = CooccurrenceRecommender()
        recommender.fit(train_df, load_cached_data=load_cached_data)

        # Generate candidates for validation users
        val_customers = val_df[USER_ID_COL].unique()

        # Check if we have cached candidates
        if load_cached_data and CACHE_CANDIDATES_TRAIN.exists():
            print("Loading cached training candidates...")
            candidates = pd.read_parquet(CACHE_CANDIDATES_TRAIN)
        else:
            candidates = recommender.generate_candidates(
                val_customers, k=TOP_K_RETRIEVAL
            )
            # Cache candidates
            os.makedirs(WORKING_DIR, exist_ok=True)
            candidates.to_parquet(CACHE_CANDIDATES_TRAIN, index=False)

        # 2. Labeling
        print("Labeling candidates...")
        # Create ground truth set for fast lookup: set of (user, item) tuples
        ground_truth = set(zip(val_df[USER_ID_COL], val_df[ITEM_ID_COL]))

        # Apply labels
        candidates_tuples = list(zip(candidates[USER_ID_COL], candidates[ITEM_ID_COL]))
        labels = [1 if t in ground_truth else 0 for t in candidates_tuples]
        candidates[TARGET_COL] = labels

        # 3. Feature Engineering
        # Load metadata
        if self.articles is None:
            self._load_and_prep_metadata()

        # Merge User Features
        print("Merging User Features...")
        candidates = candidates.merge(
            self.customers[([USER_ID_COL] + USER_FEATURES)], on=USER_ID_COL, how="left"
        )

        # Merge Item Features
        print("Merging Item Features...")
        candidates = candidates.merge(
            self.articles[([ITEM_ID_COL] + ITEM_FEATURES)], on=ITEM_ID_COL, how="left"
        )

        # Visual Features
        # Compute profiles on train_df
        self._compute_user_profiles(train_df)
        candidates["visual_similarity"] = self._compute_visual_similarity(candidates)

        # Item Popularity
        candidates["item_popularity"] = self._compute_item_popularity(
            train_df, candidates
        )

        # Save
        print(f"Saving ranker dataset to {CACHE_RANKER_DATASET}...")
        os.makedirs(WORKING_DIR, exist_ok=True)
        candidates.to_parquet(CACHE_RANKER_DATASET, index=False)

        return candidates

    def build_test_set(self, history_df, test_customer_ids, load_cached_data=True):
        """
        Builds the unlabeled test dataset for inference.

        1. Fit Retrieval on history_df (Train + Val).
        2. Generate candidates for test_customer_ids.
        3. Compute features.
        """
        print("Building Ranker Test Set...")

        # 1. Retrieval
        recommender = CooccurrenceRecommender()
        # For test set, we typically want to ensure we fit on the provided history_df
        # If load_cached_data is True, the recommender might load an old matrix.
        # However, since the matrix path is fixed in retrieval.py, we rely on the caller
        # to manage the cache state or we pass load_cached_data=False if we want to force retrain.
        # Here we follow the argument passed.
        recommender.fit(history_df, load_cached_data=load_cached_data)

        if load_cached_data and CACHE_CANDIDATES_TEST.exists():
            print("Loading cached test candidates...")
            candidates = pd.read_parquet(CACHE_CANDIDATES_TEST)
        else:
            candidates = recommender.generate_candidates(
                test_customer_ids, k=TOP_K_RETRIEVAL
            )
            os.makedirs(WORKING_DIR, exist_ok=True)
            candidates.to_parquet(CACHE_CANDIDATES_TEST, index=False)

        # 2. Feature Engineering
        if self.articles is None:
            self._load_and_prep_metadata()

        print("Merging features for Test Set...")
        candidates = candidates.merge(
            self.customers[([USER_ID_COL] + USER_FEATURES)], on=USER_ID_COL, how="left"
        )
        candidates = candidates.merge(
            self.articles[([ITEM_ID_COL] + ITEM_FEATURES)], on=ITEM_ID_COL, how="left"
        )

        # Visual Features
        # Compute profiles on full history
        self._compute_user_profiles(history_df)
        candidates["visual_similarity"] = self._compute_visual_similarity(candidates)

        # Item Popularity
        candidates["item_popularity"] = self._compute_item_popularity(
            history_df, candidates
        )

        return candidates
