import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from library.config import Config
from library.utils import set_seed, is_semiotic
from library.tokenizer import train_tokenizers
from library.hfbb import HFBBModel
from library.dataset import ContextWindowDataset


def _get_kfold_curriculum_indices(df: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """
    Generates curriculum indices using K-Fold Cross-Validation.
    This ensures 'Residuals' are true generalization failures, not just training errors.
    """
    print(f"Generating K-Fold ({n_splits}) curriculum indices...")

    # Prepare storage for indices
    residual_indices = []
    ambiguous_indices = []
    anchor_candidates = []

    # Setup K-Fold
    gkf = GroupKFold(n_splits=n_splits)

    # We need to iterate through folds
    # Note: df is the full training set
    fold = 0
    for train_idx, val_idx in gkf.split(df, groups=df["sentence_id"]):
        fold += 1
        print(f"Processing Fold {fold}/{n_splits}...")

        # 1. Train HFBB on this fold's training data
        # We must set load_cached_data=False to force re-computation for this split
        # This will overwrite the cache files temporarily, which is expected.
        fold_train_df = df.iloc[train_idx]
        hfbb = HFBBModel()
        hfbb.fit(fold_train_df, load_cached_data=False)

        # 2. Predict on this fold's validation data (the 'holdout' for this step)
        fold_val_df = df.iloc[val_idx]

        # Extract columns for vectorized processing
        # We need to handle context shifting carefully within the fold dataframe
        # However, simply extracting the rows preserves their content.
        # We need to re-compute prev/next context relative to the sentence structure.
        # Since we use GroupKFold, sentences are intact.

        # Prepare arrays for vectorized lookup
        before_series = fold_val_df["before"].astype(str).values
        after_series = fold_val_df["after"].astype(str).values
        sentence_ids = fold_val_df["sentence_id"].values

        # Vectorized Context Shifting
        # We can't rely on pandas shift easily on numpy arrays without handling boundaries
        # But we can do it on the dataframe slice before converting
        fold_val_df = fold_val_df.copy()
        fold_val_df["prev"] = fold_val_df["before"].shift(1).fillna("<start>")
        fold_val_df.loc[
            fold_val_df["sentence_id"] != fold_val_df["sentence_id"].shift(1), "prev"
        ] = "<start>"

        fold_val_df["next"] = fold_val_df["before"].shift(-1).fillna("<end>")
        fold_val_df.loc[
            fold_val_df["sentence_id"] != fold_val_df["sentence_id"].shift(-1), "next"
        ] = "<end>"

        prev_arr = fold_val_df["prev"].astype(str).values
        curr_arr = fold_val_df["before"].astype(str).values
        next_arr = fold_val_df["next"].astype(str).values
        truth_arr = fold_val_df["after"].astype(str).values

        # Access maps directly for speed
        trigram_map = hfbb.trigram_map
        bigram_prev_map = hfbb.bigram_prev_map
        bigram_next_map = hfbb.bigram_next_map
        unigram_map = hfbb.unigram_map

        # Iterate and classify
        # We need the original indices from the main dataframe 'df'
        # val_idx contains the integer positions in 'df'

        for i, real_idx in enumerate(val_idx):
            p, c, n, truth = prev_arr[i], curr_arr[i], next_arr[i], truth_arr[i]

            pred = None
            conf = 0.0

            # Hierarchy Lookup
            if (p, c, n) in trigram_map:
                pred = trigram_map[(p, c, n)]
                conf = 1.0
            elif (p, c) in bigram_prev_map:
                pred = bigram_prev_map[(p, c)]
                conf = 1.0
            elif (c, n) in bigram_next_map:
                pred = bigram_next_map[(c, n)]
                conf = 1.0
            elif c in unigram_map:
                pred, conf = unigram_map[c]

            # Logic
            if pred != truth:
                residual_indices.append(real_idx)
            else:
                if conf < Config.AMBIGUITY_THRESHOLD:
                    ambiguous_indices.append(real_idx)
                else:
                    if is_semiotic(c):
                        anchor_candidates.append(real_idx)

    print(f"K-Fold Complete.")
    print(f"  Residuals: {len(residual_indices)}")
    print(f"  Ambiguous: {len(ambiguous_indices)}")
    print(f"  Anchor Candidates: {len(anchor_candidates)}")

    # 3. Sample Anchors
    rng = np.random.default_rng(Config.SEED)
    num_anchors = int(len(anchor_candidates) * Config.ANCHOR_RATIO)
    if num_anchors > 0:
        selected_anchors = rng.choice(
            anchor_candidates, size=num_anchors, replace=False
        )
    else:
        selected_anchors = []

    print(f"  Selected Anchors: {len(selected_anchors)}")

    # 4. Combine
    base_indices = np.concatenate(
        [
            np.array(residual_indices, dtype=np.int64),
            np.array(ambiguous_indices, dtype=np.int64),
            np.array(selected_anchors, dtype=np.int64),
        ]
    )

    # 5. Upsample Rare Classes
    if Config.UPSAMPLE_RARE_CLASSES and "class" in df.columns:
        print("Applying class-balanced upsampling...")
        selected_classes = df.iloc[base_indices]["class"].values

        target_rare = {
            "MONEY",
            "MEASURE",
            "DECIMAL",
            "ORDINAL",
            "TELEPHONE",
            "ELECTRONIC",
            "DIGIT",
        }

        upsample_indices = []
        for i, cls in enumerate(selected_classes):
            if cls in target_rare:
                # Add 2 extra copies
                upsample_indices.extend([base_indices[i]] * 2)

        if upsample_indices:
            print(f"  Upsampled {len(upsample_indices)} rare tokens.")
            final_indices = np.concatenate(
                [base_indices, np.array(upsample_indices, dtype=np.int64)]
            )
        else:
            final_indices = base_indices
    else:
        final_indices = base_indices

    rng.shuffle(final_indices)
    return final_indices


def prepare_curriculum_data(load_cached_data: bool = True):
    """
    Main entry point for data preparation.

    1. Trains/Loads Tokenizers.
    2. Generates Curriculum Indices (using K-Fold CV logic).
    3. Retrains HFBB on FULL dataset (crucial for inference).
    4. Returns Datasets and Model.

    Args:
        load_cached_data (bool): Whether to use cached artifacts.

    Returns:
        tuple: (train_dataset, val_dataset, hfbb_model, char_tokenizer, target_tokenizer)
    """
    set_seed(Config.SEED)

    # 1. Load Data
    print("Loading raw metadata...")
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(Config.VAL_DATA_PATH)

    # Debug mode
    if Config.DEBUG:
        print(f"DEBUG MODE: Truncating data to {Config.DEBUG_SIZE} rows.")
        df_train = df_train.iloc[: Config.DEBUG_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SIZE]

    # 2. Tokenizers
    char_tokenizer, target_tokenizer = train_tokenizers(
        load_cached_data=load_cached_data
    )

    # 3. Curriculum Indices Generation
    indices_cache_path = os.path.join(
        Config.WORKING_DIR, "kfold_curriculum_indices.npy"
    )

    if load_cached_data and os.path.exists(indices_cache_path):
        print(f"Loading curriculum indices from {indices_cache_path}...")
        train_indices = np.load(indices_cache_path)
    else:
        # Perform K-Fold generation
        train_indices = _get_kfold_curriculum_indices(df_train, n_splits=5)
        np.save(indices_cache_path, train_indices)
        print(f"Saved curriculum indices to {indices_cache_path}")

    # 4. Final HFBB Training
    # We must ensure the HFBB model returned is trained on the FULL training set
    # for maximum accuracy during the inference phase.
    # The K-Fold loop would have left the cache in a state corresponding to the last fold.
    print("Training Final HFBB Model on full dataset...")
    hfbb_model = HFBBModel()
    hfbb_model.fit(df_train, load_cached_data=False)  # Force re-compute on full data

    # 5. Create Datasets
    print("Creating PyTorch Datasets...")

    train_dataset = ContextWindowDataset(
        df=df_train,
        indices=train_indices,
        char_tokenizer=char_tokenizer,
        target_tokenizer=target_tokenizer,
        mode="train",
    )

    # For validation, we use the full validation set (no curriculum filtering)
    val_indices = np.arange(len(df_val))
    val_dataset = ContextWindowDataset(
        df=df_val,
        indices=val_indices,
        char_tokenizer=char_tokenizer,
        target_tokenizer=target_tokenizer,
        mode="val",
    )

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")

    return train_dataset, val_dataset, hfbb_model, char_tokenizer, target_tokenizer
