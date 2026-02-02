import pandas as pd
import numpy as np
import gc
from scipy.stats import linregress
from library.config import Config
from library.embedder import LatentEmbedder


class FeatureEngineer:
    """
    Generates interaction-aware features for the ranking stage.
    Combines behavioral signals (retrieval scores), semantic signals (embeddings),
    affinity signals (user preferences), and trend signals (sales velocity).
    """

    def __init__(self):
        self.embedder = LatentEmbedder()

    def generate_features(
        self,
        candidates: pd.DataFrame,
        transactions: pd.DataFrame,
        customers: pd.DataFrame,
        articles: pd.DataFrame,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Main method to generate features. Handles caching.
        """
        # Define cache key based on inputs
        max_date = transactions["t_dat"].max().date()
        params = {
            "n_candidates": len(candidates),
            "max_date": str(max_date),
            "val_days": Config.VAL_SIZE_DAYS,
            "trend_window": Config.POPULARITY_WINDOW_DAYS,
        }
        cache_path = Config.get_cache_path("features_ranker.parquet", params)

        # 1. Try to load from cache
        if load_cached_data and cache_path.exists():
            print(f"Loading cached features from {cache_path}")
            return pd.read_parquet(cache_path)

        print("Generating features from scratch...")

        # Ensure article_id consistency
        candidates["article_id"] = candidates["article_id"].astype(str)
        articles["article_id"] = articles["article_id"].astype(str)
        transactions["article_id"] = transactions["article_id"].astype(str)

        # 2. Merge Metadata
        print("Merging metadata...")
        # Merge Customers
        cust_cols = [
            "customer_id",
            "age",
            "club_member_status",
            "fashion_news_frequency",
        ]
        # Handle missing columns in customers if any (robustness)
        cust_cols = [c for c in cust_cols if c in customers.columns]
        df = candidates.merge(customers[cust_cols], on="customer_id", how="left")

        # Merge Articles
        art_cols = [
            "article_id",
            "product_type_no",
            "graphical_appearance_no",
            "colour_group_code",
            "department_no",
            "index_group_no",
            "section_no",
            "garment_group_no",
        ]
        art_cols = [c for c in art_cols if c in articles.columns]
        df = df.merge(articles[art_cols], on="article_id", how="left")

        # 3. Compute Embedding Similarity (Dense Feature)
        print("Computing dense embedding similarities...")
        df = self._add_embedding_similarity(df, transactions)

        # 4. Compute Affinity Features
        print("Computing user-category affinity features...")
        df = self._add_affinity_features(df, transactions, articles)

        # 5. Compute Trend Features
        print("Computing trend (sales velocity) features...")
        df = self._add_trend_features(df, transactions)

        # 6. Compute Last Item Similarity (Sequential Signal)
        print("Computing last item similarity features...")
        df = self._add_last_item_similarity(df, transactions)

        # 7. Final Cleanup
        # Convert categoricals to appropriate types if needed, or leave for LGBM to handle
        # Fill NaNs in numerical features
        fill_zero_cols = [
            "cooc_score",
            "embed_score",
            "repur_count",
            "pop_flag",
            "user_dept_affinity",
            "user_index_affinity",
            "sales_velocity",
            "sales_count_7d",
            "last_item_embed_sim",
        ]
        for col in fill_zero_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Fill NaNs in metadata (e.g. age) with mean or mode
        if "age" in df.columns:
            df["age"] = df["age"].fillna(df["age"].mean())

        # Encode categorical strings as integers/categories for LightGBM
        cat_cols = ["club_member_status", "fashion_news_frequency"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        print(f"Feature generation complete. Shape: {df.shape}")

        # 7. Save to cache
        df.to_parquet(cache_path, index=False)

        return df

    def _add_embedding_similarity(
        self, df: pd.DataFrame, transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes the Cosine Similarity between the User Vector and Item Vector for all candidates.
        This fills in the 'embed_score' for candidates retrieved via Co-occurrence/Popularity.
        """
        # Ensure embedder is trained/loaded
        self.embedder.fit(transactions)

        if self.embedder.embeddings is None:
            df["dense_embed_score"] = 0.0
            return df

        # Get User Embeddings for users in the candidate set
        unique_users = df["customer_id"].unique()
        # Filter transactions to relevant history for these users to speed up calculation
        # (LatentEmbedder.get_user_embeddings handles the grouping)
        # We pass the full transactions or a subset. Passing full is safer for complete history.
        user_emb_map = self.embedder.get_user_embeddings(
            transactions[transactions["customer_id"].isin(unique_users)]
        )

        # Get Item Embeddings
        # We can access self.embedder.get_embedding(aid)

        # Vectorized calculation
        # 1. Map users to vectors
        # Create a temp column with user vectors
        # Note: This might be memory intensive if we put arrays in cells.
        # Instead, we process in batches or use a lookup.

        # Efficient approach:
        # Extract indices for users and items

        # Map customer_id to a temporary index 0..N
        u_map = {u: i for i, u in enumerate(user_emb_map.keys())}
        # Create matrix of user vectors
        if not u_map:
            df["dense_embed_score"] = 0.0
            return df

        u_matrix = np.array(list(user_emb_map.values()))
        # Normalize user matrix
        u_norm = np.linalg.norm(u_matrix, axis=1, keepdims=True)
        u_norm[u_norm == 0] = 1e-10
        u_matrix = u_matrix / u_norm

        # Map article_id to embedding index
        # self.embedder.vocab maps article_id -> index in self.embedder.embeddings
        vocab = self.embedder.vocab
        embeddings = self.embedder.embeddings

        # Normalize item embeddings globally
        i_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        i_norm[i_norm == 0] = 1e-10
        embeddings_norm = embeddings / i_norm

        # Function to compute score row-wise (or use apply)
        # To be fast, we can use the fact that we have user_id and article_id.

        # Let's use a dictionary lookup for speed
        # user_id -> vector (normalized)
        u_dict = {u: vec for u, vec in zip(user_emb_map.keys(), u_matrix)}

        # article_id -> vector (normalized)
        # Pre-fetch only needed articles
        needed_articles = df["article_id"].unique()
        i_dict = {}
        for aid in needed_articles:
            if aid in vocab:
                idx = vocab[aid]
                i_dict[aid] = embeddings_norm[idx]

        # Compute dot product
        def compute_sim(row):
            u_vec = u_dict.get(row["customer_id"])
            i_vec = i_dict.get(row["article_id"])
            if u_vec is None or i_vec is None:
                return 0.0
            return float(np.dot(u_vec, i_vec))

        # Apply is slow but robust. For 10M rows it might take a few minutes.
        # Optimization: Use numpy arrays aligned with DataFrame

        # Map IDs to indices in our local matrices
        # This is complex to implement efficiently without huge memory.
        # Let's stick to a slightly optimized apply or map.

        # Actually, we can update the 'embed_score' column where it is 0.
        # But let's create a new column 'dense_embed_score'.

        # Optimized:
        # 1. Get arrays of vectors aligned with DF
        # This requires mapping.

        # Define a default zero vector
        dim = Config.EMBED_DIM
        zero_vec = np.zeros(dim, dtype=np.float32)

        # We can't easily vectorize the map -> stack process for 10M rows without memory spike.
        # We will use the existing 'embed_score' from retrieval as a base,
        # but retrieval only has top K.
        # Let's try to compute it.

        print("  Calculating dot products...")
        # We'll use a loop over chunks to manage memory
        chunk_size = 100000
        scores = []

        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            chunk = df.iloc[start:end]

            # Get user vectors
            u_ids = chunk["customer_id"].values
            i_ids = chunk["article_id"].values

            # Build batch matrices
            # Using list comprehension is faster than apply
            u_vecs = [u_dict.get(u, zero_vec) for u in u_ids]
            i_vecs = [i_dict.get(i, zero_vec) for i in i_ids]

            u_vecs = np.array(u_vecs)
            i_vecs = np.array(i_vecs)

            # Dot product
            # (N, D) * (N, D) -> (N,) sum over axis 1
            batch_scores = np.sum(u_vecs * i_vecs, axis=1)
            scores.append(batch_scores)

        df["dense_embed_score"] = np.concatenate(scores)

        return df

    def _add_affinity_features(
        self, df: pd.DataFrame, transactions: pd.DataFrame, articles: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates user affinity towards item categories (Department, Index Group).
        Feature: 'user_dept_affinity' = (User's purchases in Dept X) / (User's total purchases)
        """
        # 1. Prepare User Profiles
        # Merge transactions with article metadata to get categories
        # Use a subset of columns to save memory
        hist_df = transactions[["customer_id", "article_id"]].merge(
            articles[["article_id", "department_no", "index_group_no"]],
            on="article_id",
            how="left",
        )

        # Calculate Department Affinity
        # Count user purchases per department
        dept_counts = (
            hist_df.groupby(["customer_id", "department_no"])
            .size()
            .reset_index(name="dept_count")
        )
        # Count total user purchases
        user_totals = (
            hist_df.groupby("customer_id").size().reset_index(name="user_total")
        )

        dept_affinity = dept_counts.merge(user_totals, on="customer_id")
        dept_affinity["user_dept_affinity"] = (
            dept_affinity["dept_count"] / dept_affinity["user_total"]
        )

        # Calculate Index Group Affinity
        index_counts = (
            hist_df.groupby(["customer_id", "index_group_no"])
            .size()
            .reset_index(name="index_count")
        )
        index_affinity = index_counts.merge(user_totals, on="customer_id")
        index_affinity["user_index_affinity"] = (
            index_affinity["index_count"] / index_affinity["user_total"]
        )

        # 2. Map to Candidates
        # Candidates already have 'department_no' and 'index_group_no' from the metadata merge in generate_features

        # Merge Dept Affinity
        df = df.merge(
            dept_affinity[["customer_id", "department_no", "user_dept_affinity"]],
            on=["customer_id", "department_no"],
            how="left",
        )

        # Merge Index Affinity
        df = df.merge(
            index_affinity[["customer_id", "index_group_no", "user_index_affinity"]],
            on=["customer_id", "index_group_no"],
            how="left",
        )

        return df

    def _add_last_item_similarity(
        self, df: pd.DataFrame, transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates similarity between the candidate item and the user's most recent purchase.
        This captures immediate sequential intent (e.g., bought trousers -> buy belt).
        """
        # 1. Identify Last Item per User
        # Sort by date and take the last one
        last_items = (
            transactions.sort_values("t_dat")
            .groupby("customer_id")
            .tail(1)[["customer_id", "article_id"]]
        )
        last_items = last_items.rename(columns={"article_id": "last_article_id"})

        # 2. Merge into Candidates
        # We only need to compute this for users who have a history
        df = df.merge(last_items, on="customer_id", how="left")

        # 3. Compute Embedding Similarity
        # Reuse the embedder logic
        if self.embedder.embeddings is None:
            df["last_item_embed_sim"] = 0.0
            return df.drop(columns=["last_article_id"], errors="ignore")

        vocab = self.embedder.vocab
        embeddings = self.embedder.embeddings

        # Normalize embeddings globally
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embeddings_norm = embeddings / norms

        # Create Lookup Dictionary
        # article_id -> normalized vector
        # We only need vectors for items present in 'article_id' (candidate) or 'last_article_id'
        needed_items = set(df["article_id"].unique()) | set(
            df["last_article_id"].dropna().unique()
        )

        vec_dict = {}
        for aid in needed_items:
            if aid in vocab:
                idx = vocab[aid]
                vec_dict[aid] = embeddings_norm[idx]

        # Define Zero Vector
        dim = Config.EMBED_DIM
        zero_vec = np.zeros(dim, dtype=np.float32)

        # Compute Dot Product
        # Chunked approach to avoid OOM on large datasets
        print("  Calculating last item similarity in chunks...")
        chunk_size = 100000
        sims = []

        cand_ids_all = df["article_id"].values
        last_ids_all = df["last_article_id"].values

        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))

            # Get chunk IDs
            c_ids = cand_ids_all[start:end]
            l_ids = last_ids_all[start:end]

            # Map to vectors
            c_vecs = np.array([vec_dict.get(i, zero_vec) for i in c_ids])
            l_vecs = np.array([vec_dict.get(i, zero_vec) for i in l_ids])

            # Dot product
            batch_sims = np.sum(c_vecs * l_vecs, axis=1)
            sims.append(batch_sims)

        df["last_item_embed_sim"] = np.concatenate(sims)

        # Cleanup temp column
        df = df.drop(columns=["last_article_id"])

        return df

    def _add_trend_features(
        self, df: pd.DataFrame, transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates sales velocity (slope) for the last 7 days.
        """
        max_date = transactions["t_dat"].max()
        start_date = max_date - pd.Timedelta(days=Config.POPULARITY_WINDOW_DAYS)

        # Filter last 7 days
        recent = transactions[transactions["t_dat"] > start_date].copy()

        # Count daily sales per article
        # Group by article and date
        daily_counts = (
            recent.groupby(["article_id", "t_dat"]).size().reset_index(name="count")
        )

        # We need to calculate slope for each article.
        # Map dates to integers 0..6
        dates = sorted(daily_counts["t_dat"].unique())
        date_map = {d: i for i, d in enumerate(dates)}
        daily_counts["day_idx"] = daily_counts["t_dat"].map(date_map)

        # Calculate slope
        # We can use a simplified formula or apply.
        # Since we have max 7 points per article, we can pivot.

        pivot = daily_counts.pivot(
            index="article_id", columns="day_idx", values="count"
        ).fillna(0)

        # Ensure all 7 columns exist (0 to 6)
        for i in range(Config.POPULARITY_WINDOW_DAYS):
            if i not in pivot.columns:
                pivot[i] = 0

        # Sort columns
        pivot = pivot.sort_index(axis=1)

        X = np.arange(Config.POPULARITY_WINDOW_DAYS)
        # Slope formula: (N * sum(xy) - sum(x)sum(y)) / (N * sum(x^2) - sum(x)^2)
        # Precompute X constants
        N = len(X)
        sum_x = np.sum(X)
        sum_x_sq = np.sum(X**2)
        denom = N * sum_x_sq - sum_x**2

        # Y is the row values
        Y = pivot.values
        sum_y = np.sum(Y, axis=1)
        sum_xy = np.sum(Y * X, axis=1)

        slopes = (N * sum_xy - sum_x * sum_y) / denom

        # Create feature dataframe
        trend_df = pd.DataFrame(
            {
                "article_id": pivot.index,
                "sales_velocity": slopes,
                "sales_count_7d": sum_y,  # Total sales in last 7 days
            }
        )

        # Merge back
        df = df.merge(trend_df, on="article_id", how="left")

        return df
