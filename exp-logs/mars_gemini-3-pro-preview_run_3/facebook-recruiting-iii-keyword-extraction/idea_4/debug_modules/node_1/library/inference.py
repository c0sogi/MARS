import os
import torch
import pandas as pd
import numpy as np
import library.config as config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import HybridCNNTransformer


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set and saves them to a submission file.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
                                 Defaults to True.
    """
    # 1. Setup
    set_seed(config.SEED)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    print("Loading data for inference...")
    # We ignore train/val loaders here, only need test_loader and artifacts
    _, _, test_loader, tokenizer, encoder = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model Architecture
    # Must match the architecture used in training
    print("Initializing model...")
    vocab_size = len(tokenizer.vocab) + 1  # +1 for safety/consistency with trainer
    num_classes = len(encoder.classes_)

    model = HybridCNNTransformer(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        cnn_filters=config.CNN_FILTERS,
        cnn_kernel_size=config.CNN_KERNEL_SIZE,
        transformer_layers=config.TRANSFORMER_LAYERS,
        num_heads=config.NUM_HEADS,
        transformer_ff_dim=config.TRANSFORMER_FF_DIM,
        dropout=config.DROPOUT,
        num_classes=num_classes,
        max_len=config.MAX_LEN,
    )

    # 3. Load Trained Weights
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {config.MODEL_SAVE_PATH}. Train the model first."
        )

    print(f"Loading weights from {config.MODEL_SAVE_PATH}...")
    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    model.load_state_dict(state_dict)

    model.to(config.DEVICE)
    model.eval()

    # 4. Inference Loop
    print("Running inference on test set...")
    all_ids = []
    all_pred_tags = []

    # Pre-convert classes to numpy array for fast indexing
    class_names = np.array(encoder.classes_)

    with torch.no_grad():
        for tokens, ids in test_loader:
            tokens = tokens.to(config.DEVICE)

            # Forward pass
            logits = model(tokens)
            probs = torch.sigmoid(logits)

            # Move to CPU
            probs = probs.cpu().numpy()
            ids = ids.numpy()

            # Apply Threshold
            preds_binary = probs >= config.PREDICTION_THRESHOLD

            # Decode Tags
            # Iterate over the batch
            for i in range(len(ids)):
                # Get boolean mask for this sample
                row_mask = preds_binary[i]

                # Select tags where prediction is True
                predicted_tags = class_names[row_mask]

                # Join into space-delimited string
                tag_str = " ".join(predicted_tags)

                all_ids.append(ids[i])
                all_pred_tags.append(tag_str)

    # 5. Create Submission DataFrame
    print("Creating submission file...")
    df_submission = pd.DataFrame({"Id": all_ids, "Tags": all_pred_tags})

    # 6. Save
    df_submission.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")
    print(f"Total predictions: {len(df_submission)}")
