import os
import torch
import pandas as pd
from library.config import Config
from library.model import AttributeContextualizedTransformer
from library.dataset import get_dataloaders


def greedy_decode(model, images, max_len=None):
    """
    Performs autoregressive greedy decoding using the trained model's decoder.

    Args:
        model (AttributeContextualizedTransformer): The trained model.
        images (torch.Tensor): Batch of images (B, C, H, W).
        max_len (int, optional): Maximum sequence length. Defaults to Config.MAX_LEN.

    Returns:
        torch.Tensor: Predicted sequences of indices (B, Seq_Len).
    """
    # The model class implements the autoregressive loop in its predict method.
    return model.predict(images, max_len=max_len)


def generate_submission(
    model_path=None, output_path=None, load_cached_data=True, debug=False
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model_path (str, optional): Path to the trained model weights. Defaults to Config.MODEL_SAVE_PATH.
        output_path (str, optional): Path to save the submission CSV. Defaults to Config.SUBMISSION_PATH.
        load_cached_data (bool): Whether to use cached metadata/dataloaders/outputs.
        debug (bool): If True, runs on a subset of the test data.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    # Set defaults if not provided
    if model_path is None:
        model_path = Config.MODEL_SAVE_PATH
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Caching Logic: If the output file exists and caching is enabled, load and return it.
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading existing submission from {output_path}")
        return pd.read_csv(output_path)

    print("Starting submission generation...")

    # 1. Setup Device
    device = torch.device(Config.DEVICE)
    print(f"Inference Device: {device}")

    # 2. Load DataLoaders and Tokenizer
    # get_dataloaders handles metadata caching internally.
    # We unpack the tuple to get the test_loader and tokenizer.
    _, _, test_loader, tokenizer = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 3. Initialize Model
    print("Initializing model...")
    model = AttributeContextualizedTransformer()
    model.to(device)

    # 4. Load Model Weights
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using random weights (for debugging/testing flow)."
        )

    # 5. Inference Loop
    model.eval()
    results = []

    print(f"Processing {len(test_loader)} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            image_ids = batch["image_id"]

            # Run greedy decoding
            pred_seqs = greedy_decode(model, images, max_len=Config.MAX_LEN)

            # Decode sequences to text
            # pred_seqs is a tensor of indices
            pred_texts = [tokenizer.sequence_to_text(s) for s in pred_seqs]

            # Collect results
            for img_id, text in zip(image_ids, pred_texts):
                results.append({"image_id": img_id, "InChI": text})

            if debug and batch_idx >= 5:
                print("Debug mode: stopping after 5 batches.")
                break

    # 6. Create DataFrame and Save
    submission_df = pd.DataFrame(results)

    # Ensure columns are in correct order
    submission_df = submission_df[["image_id", "InChI"]]

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions generated: {len(submission_df)}")

    return submission_df
