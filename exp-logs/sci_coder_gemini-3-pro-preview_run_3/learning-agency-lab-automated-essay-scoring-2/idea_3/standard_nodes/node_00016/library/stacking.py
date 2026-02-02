import os
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from library.config import Config
from library.data import process_data, tokenize_sliding_window
from library.utils import compute_qwk


def get_feature_cols(df):
    """
    Identifies meta-feature columns in the dataframe.
    """
    non_feature_cols = {"essay_id", "full_text", "score", "source_file"}
    return [c for c in df.columns if c not in non_feature_cols]


def extract_features(df, model, tokenizer, device, split_name, load_cached_data=True):
    """
    Extracts embeddings using the fine-tuned backbone and combines them with meta-features.
    Implements caching mechanism to avoid re-computing embeddings.

    Args:
        df (pd.DataFrame): Dataframe containing text and meta-features.
        model (nn.Module): Fine-tuned EssayModel.
        tokenizer: Transformer tokenizer.
        device: Torch device.
        split_name (str): 'train', 'val', or 'test' for cache naming.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (features_array, labels_array, ids_array)
    """
    # Define cache paths
    cache_embed_path = os.path.join(Config.CACHE_DIR, f"{split_name}_embeddings.npy")
    cache_meta_path = os.path.join(Config.CACHE_DIR, f"{split_name}_meta.npy")
    cache_ids_path = os.path.join(Config.CACHE_DIR, f"{split_name}_ids.npy")
    cache_labels_path = os.path.join(Config.CACHE_DIR, f"{split_name}_labels.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if os.path.exists(cache_embed_path) and os.path.exists(cache_meta_path):
            print(f"Loading cached features for {split_name}...")
            embeddings = np.load(cache_embed_path)
            meta_features = np.load(cache_meta_path)
            ids = np.load(cache_ids_path, allow_pickle=True)

            labels = None
            if os.path.exists(cache_labels_path):
                labels = np.load(cache_labels_path)

            # Combine embeddings and meta-features
            features = np.concatenate([embeddings, meta_features], axis=1)
            return features, labels, ids

    # 2. Compute Features
    print(f"Extracting features for {split_name}...")

    model.eval()
    model.to(device)

    embeddings_list = []
    meta_features_list = []
    ids_list = []
    labels_list = []

    # Identify meta-feature columns
    feature_cols = get_feature_cols(df)

    # Pre-extract data to avoid pandas overhead in loop
    texts = df["full_text"].astype(str).values
    essay_ids = df["essay_id"].values
    meta_vals = df[feature_cols].values.astype(np.float32)

    has_labels = "score" in df.columns
    if has_labels:
        scores = df["score"].values.astype(np.float32)

    with torch.no_grad():
        for i in range(len(df)):
            text = texts[i]

            # Sliding Window Tokenization
            # Returns dict with tensors of shape [num_chunks, seq_len]
            inputs = tokenize_sliding_window(text, tokenizer)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Model Inference (Process all chunks for this essay)
            with torch.amp.autocast("cuda", enabled=Config.use_amp):
                outputs = model(input_ids, attention_mask)
                # outputs["embeddings"] shape: [num_chunks, hidden_size]
                chunk_embeddings = outputs["embeddings"]

            # Average Pooling over chunks to get single vector per essay
            # shape: [hidden_size]
            avg_embedding = torch.mean(chunk_embeddings, dim=0).cpu().numpy()

            embeddings_list.append(avg_embedding)
            meta_features_list.append(meta_vals[i])
            ids_list.append(essay_ids[i])

            if has_labels:
                labels_list.append(scores[i])

    # Convert to numpy arrays
    embeddings = np.array(embeddings_list, dtype=np.float32)
    meta_features = np.array(meta_features_list, dtype=np.float32)
    ids = np.array(ids_list)
    labels = np.array(labels_list, dtype=np.float32) if has_labels else None

    # 3. Save to Cache
    print(f"Saving features to {Config.CACHE_DIR}...")
    np.save(cache_embed_path, embeddings)
    np.save(cache_meta_path, meta_features)
    np.save(cache_ids_path, ids)
    if labels is not None:
        np.save(cache_labels_path, labels)

    # Combine
    features = np.concatenate([embeddings, meta_features], axis=1)

    return features, labels, ids


def train_lgbm(train_feats, train_labels, val_feats, val_labels):
    """
    Trains a LightGBM regressor on the combined features.

    Args:
        train_feats: Training features (embeddings + meta).
        train_labels: Training scores.
        val_feats: Validation features.
        val_labels: Validation scores.

    Returns:
        model: Trained LightGBM model.
    """
    print("Training LightGBM Stacking Model...")

    train_data = lgb.Dataset(train_feats, label=train_labels)
    val_data = lgb.Dataset(val_feats, label=val_labels, reference=train_data)

    # Setup callbacks for logging and early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    # Train
    model = lgb.train(
        Config.lgbm_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        num_boost_round=Config.lgbm_params["n_estimators"],
        callbacks=callbacks,
    )

    # Validation Evaluation
    val_preds = model.predict(val_feats)
    val_qwk = compute_qwk(val_labels, val_preds)
    print(f"LightGBM Validation QWK: {val_qwk}")

    return model


def predict_stacking(model, test_feats, test_ids):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: Trained LightGBM model.
        test_feats: Test features.
        test_ids: Test essay IDs.
    """
    print("Generating predictions...")
    preds = model.predict(test_feats)

    # Clip predictions to valid range [1, 6] and round to nearest integer
    final_preds = np.clip(preds, 1, 6).round().astype(int)

    # Create DataFrame
    submission = pd.DataFrame({"essay_id": test_ids, "score": final_preds})

    # Save submission
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
    print(submission.head())
