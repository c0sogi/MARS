import os
import torch
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.model import BiGRUClassifier
from library.data_loader import TextTokenizer, TagEncoder, get_test_dataloader


def run_prediction(debug=Config.DEBUG):
    """
    Loads the trained model and artifacts, runs inference on the test set,
    and generates the submission file.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data.
    """
    print("--- Starting Prediction Pipeline ---")

    # 1. Set Seed and Device
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Load Artifacts (Tokenizer and TagEncoder)
    # We assume training has already run, so caches exist.
    # We pass empty containers because we rely on loading from cache.
    print("Loading tokenizer...")
    tokenizer = TextTokenizer()
    try:
        # Pass empty list; load_or_create will ignore it if cache exists
        tokenizer.fit(texts=[], load_cached_data=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer cache. Ensure training is run first. Error: {e}"
        )

    if tokenizer.vocab is None:
        raise RuntimeError("Tokenizer vocabulary not loaded.")

    print("Loading tag encoder...")
    tag_encoder = TagEncoder()
    try:
        # Pass empty series; load_or_create will ignore it if cache exists
        tag_encoder.fit(
            tags_series=pd.Series([], dtype="object"), load_cached_data=True
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tag encoder cache. Ensure training is run first. Error: {e}"
        )

    if tag_encoder.tag_list is None:
        raise RuntimeError("TagEncoder tag list not loaded.")

    # 3. Load Test Data
    test_loader = get_test_dataloader(tokenizer, debug=debug)

    # 4. Initialize and Load Model
    vocab_size = len(tokenizer.vocab)
    print(f"Initializing model with vocab size: {vocab_size}")

    model = BiGRUClassifier(
        vocab_size=vocab_size,
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        num_classes=Config.NUM_TAGS,
        dropout=Config.DROPOUT,
    )

    model.to(device)

    print(f"Loading model checkpoint from {Config.MODEL_SAVE_PATH}...")
    try:
        load_checkpoint(model, path=Config.MODEL_SAVE_PATH, device=device)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Train the model first."
        )

    # 5. Inference Loop
    model.eval()
    all_ids = []
    all_probs = []

    print("Running inference...")
    with torch.no_grad():
        for inputs, row_ids in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)

            # Sigmoid for multi-label probabilities
            probs = torch.sigmoid(logits)

            # Store results (move to CPU to save GPU memory)
            all_probs.append(probs.cpu().numpy())
            # row_ids is a tensor of ints
            all_ids.extend(row_ids.numpy())

    # Concatenate probabilities
    if len(all_probs) == 0:
        print("Warning: No data found in test loader.")
        return

    all_probs = np.vstack(all_probs)

    # 6. Convert Probabilities to Tags
    print("Converting probabilities to tags...")
    predicted_tags = tag_encoder.inverse_transform(
        all_probs, threshold=Config.PREDICTION_THRESHOLD
    )

    # 7. Create Submission DataFrame
    print("Creating submission file...")
    submission_df = pd.DataFrame({"Id": all_ids, "Tags": predicted_tags})

    # Ensure Id is int
    submission_df["Id"] = submission_df["Id"].astype(int)

    # Sort by Id
    submission_df = submission_df.sort_values("Id").reset_index(drop=True)

    # 8. Save Submission
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(f"Submission shape: {submission_df.shape}")
    print("Head of submission:")
    print(submission_df.head())
