import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from library.config import Config


def get_data(load_cached_data=True):
    """
    Main entry point for the data pipeline.
    Loads raw data, performs feature engineering, applies scaling, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files
                                 from the cache directory.

    Returns:
        tuple: (train_x, train_y, val_x, val_y, test_x, test_ids)
               All x/y arrays are float32 tensors of shape (N, SEQ_LEN, Features) or (N, SEQ_LEN).
               test_ids is int32 array of shape (N, SEQ_LEN).
    """
    # Define cache paths
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    file_map = {
        "train_x": os.path.join(cache_dir, "train_x.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_x": os.path.join(cache_dir, "val_x.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_x": os.path.join(cache_dir, "test_x.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # 1. Try Loading Cache
    if load_cached_data:
        if all(os.path.exists(p) for p in file_map.values()):
            print(f"Loading cached data from {cache_dir}...")
            try:
                data = {k: np.load(p) for k, p in file_map.items()}
                return (
                    data["train_x"],
                    data["train_y"],
                    data["val_x"],
                    data["val_y"],
                    data["test_x"],
                    data["test_ids"],
                )
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")
        else:
            print("Cache incomplete or missing. Starting processing pipeline...")

    # 2. Load Raw Metadata
    print("Loading raw CSV files...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # 3. Handle Debug Mode (Subsampling)
    if Config.DEBUG:
        print(
            f"DEBUG MODE: Subsampling to {Config.DEBUG_SAMPLE_SIZE} breaths per split."
        )

        def subsample(df):
            breaths = df[Config.BREATH_ID_COL].unique()[: Config.DEBUG_SAMPLE_SIZE]
            return df[df[Config.BREATH_ID_COL].isin(breaths)].copy()

        train_df = subsample(train_df)
        val_df = subsample(val_df)
        test_df = subsample(test_df)

    # 4. Feature Engineering Helper (Vectorized)
    def process_split(df, is_test=False):
        """
        Transforms a DataFrame into 3D numpy arrays (N_breaths, 80, Features).
        Performs physical integration, differentiation, and lookahead ops.
        """
        # Ensure data is sorted by breath and time
        df = df.sort_values([Config.BREATH_ID_COL, Config.ID_COL])

        # Calculate dimensions
        n_samples = len(df)
        seq_len = Config.SEQ_LEN
        n_breaths = n_samples // seq_len

        # Sanity check
        if n_samples % seq_len != 0:
            raise ValueError(
                f"Data length {n_samples} is not divisible by SEQ_LEN {seq_len}"
            )

        # Extract raw columns and reshape to (N_breaths, SEQ_LEN)
        # Using .values.reshape allows for fast vectorized operations per breath
        u_in = df["u_in"].values.reshape(n_breaths, seq_len)
        u_out = df["u_out"].values.reshape(n_breaths, seq_len)
        time_step = df["time_step"].values.reshape(n_breaths, seq_len)
        R = df["R"].values.reshape(n_breaths, seq_len)
        C = df["C"].values.reshape(n_breaths, seq_len)

        # --- Feature Engineering ---

        # 1. Time Delta (dt)
        # Calculate diff along axis 1. Prepend the first time value to maintain shape?
        # Actually, dt[0] is usually 0 or t[0]. Since we integrate, let's assume dt[0]=0.
        # np.diff returns shape (N, 79). We prepend 0 to get (N, 80).
        # Note: time_step is cumulative time.
        dt = np.diff(time_step, axis=1, prepend=time_step[:, 0:1])
        dt[:, 0] = 0.0  # Force first dt to 0

        # 2. Physical Integration (Area)
        # Area = Cumulative Sum of (u_in * dt)
        area = np.cumsum(u_in * dt, axis=1)

        # 3. Derivatives (Finite Difference)
        # u_in_diff = u_in[t] - u_in[t-1]
        u_in_diff = np.diff(u_in, axis=1, prepend=u_in[:, 0:1])
        u_in_diff[:, 0] = 0.0

        # 4. Lookahead Features (Future Context)
        # We shift the array to the left (negative roll).
        # We must pad the end with 0s (assuming valve closes or no info).
        def get_lookahead(arr, steps):
            shifted = np.roll(arr, -steps, axis=1)
            shifted[:, -steps:] = 0.0
            return shifted

        u_in_next1 = get_lookahead(u_in, 1)
        u_in_next2 = get_lookahead(u_in, 2)
        u_in_next3 = get_lookahead(u_in, 3)
        u_in_next4 = get_lookahead(u_in, 4)
        u_in_diff_next1 = get_lookahead(u_in_diff, 1)

        # 5. Physics Interactions
        R_u_in = R * u_in
        area_C = area / C

        # --- Feature Assembly ---
        # Map feature names to arrays. Must match Config.FEATURE_COLS order exactly.
        feature_map = {
            "time_step": time_step,
            "u_in": u_in,
            "u_out": u_out,
            "area": area,
            "u_in_diff": u_in_diff,
            "u_in_next1": u_in_next1,
            "u_in_next2": u_in_next2,
            "u_in_next3": u_in_next3,
            "u_in_next4": u_in_next4,
            "u_in_diff_next1": u_in_diff_next1,
            "R": R,
            "C": C,
            "R__u_in": R_u_in,
            "area__C": area_C,
        }

        # Stack features into (N_breaths, SEQ_LEN, N_features)
        feature_list = [feature_map[col] for col in Config.FEATURE_COLS]
        X = np.stack(feature_list, axis=-1)

        # Handle Targets and IDs
        y = None
        ids = None

        if not is_test:
            y = df[Config.TARGET_COL].values.reshape(n_breaths, seq_len)
        else:
            ids = df[Config.ID_COL].values.reshape(n_breaths, seq_len)

        return X, y, ids

    print("Processing Training Data...")
    train_x, train_y, _ = process_split(train_df, is_test=False)

    print("Processing Validation Data...")
    val_x, val_y, _ = process_split(val_df, is_test=False)

    print("Processing Test Data...")
    test_x, _, test_ids = process_split(test_df, is_test=True)

    # 5. Robust Scaling
    print("Applying RobustScaler...")
    # Reshape to 2D (samples, features) for sklearn
    N_train, L, F = train_x.shape

    train_x_flat = train_x.reshape(-1, F)
    val_x_flat = val_x.reshape(-1, F)
    test_x_flat = test_x.reshape(-1, F)

    scaler = RobustScaler()
    # Fit only on training data to prevent leakage
    scaler.fit(train_x_flat)

    train_x_flat = scaler.transform(train_x_flat)
    val_x_flat = scaler.transform(val_x_flat)
    test_x_flat = scaler.transform(test_x_flat)

    # Reshape back to 3D
    train_x = train_x_flat.reshape(N_train, L, F)
    val_x = val_x_flat.reshape(val_x.shape[0], L, F)
    test_x = test_x_flat.reshape(test_x.shape[0], L, F)

    # 6. Save to Cache
    print("Saving processed data to cache...")
    np.save(file_map["train_x"], train_x.astype(np.float32))
    np.save(file_map["train_y"], train_y.astype(np.float32))
    np.save(file_map["val_x"], val_x.astype(np.float32))
    np.save(file_map["val_y"], val_y.astype(np.float32))
    np.save(file_map["test_x"], test_x.astype(np.float32))
    np.save(file_map["test_ids"], test_ids.astype(np.int32))

    print("Data pipeline complete.")
    return (
        train_x.astype(np.float32),
        train_y.astype(np.float32),
        val_x.astype(np.float32),
        val_y.astype(np.float32),
        test_x.astype(np.float32),
        test_ids.astype(np.int32),
    )
