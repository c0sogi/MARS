import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import InChiDataset, get_transforms
from library.model import ResNetTCN


def greedy_decode(model, images, tokenizer, max_len=None, device=None):
    """
    Performs batched greedy decoding for the ResNet-TCN model.

    Args:
        model (nn.Module): The trained ResNetTCN model.
        images (torch.Tensor): Batch of images (B, C, H, W).
        tokenizer (Tokenizer): Tokenizer instance.
        max_len (int, optional): Maximum sequence length. Defaults to Config.MAX_LEN.
        device (torch.device, optional): Device to run inference on. Defaults to Config.DEVICE.

    Returns:
        list[str]: List of predicted InChI strings.
    """
    if max_len is None:
        max_len = Config.MAX_LEN
    if device is None:
        device = Config.DEVICE

    model.eval()
    batch_size = images.size(0)

    # 1. Encode images
    # Output: (B, 512)
    with torch.no_grad():
        features = model.encoder(images)

    # 2. Initialize sequences with SOS token
    # Shape: (B, 1)
    seqs = torch.full(
        (batch_size, 1), tokenizer.SOS_IDX, dtype=torch.long, device=device
    )

    # Track finished sequences
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # 3. Autoregressive loop
    # TCN requires the full history at each step because the provided implementation is stateless
    for _ in range(max_len):
        with torch.no_grad():
            # Embed current sequence: (B, L, Emb_Dim)
            embeddings = model.embedding(seqs)

            # Expand image features to match sequence length: (B, L, Enc_Dim)
            features_repeated = features.unsqueeze(1).expand(-1, embeddings.size(1), -1)

            # Concatenate: (B, L, Emb_Dim + Enc_Dim)
            tcn_input = torch.cat((embeddings, features_repeated), dim=2)

            # Permute for Conv1d: (B, Channels, L)
            tcn_input = tcn_input.permute(0, 2, 1)

            # Forward pass through TCN
            output = model.tcn(tcn_input)

            # Get output for the last time step: (B, Channels)
            last_output = output[:, :, -1]

            # Decode to logits: (B, Vocab)
            logits = model.decoder(last_output)

            # Greedy selection
            next_tokens = torch.argmax(logits, dim=1)  # (B,)

            # Append next tokens to sequence
            seqs = torch.cat([seqs, next_tokens.unsqueeze(1)], dim=1)

            # Update finished status
            is_eos = next_tokens == tokenizer.EOS_IDX
            finished = finished | is_eos

            # If all sequences in batch have hit EOS, stop early
            if finished.all():
                break

    # 4. Convert sequences to text
    result_strings = []
    # Move to CPU for string conversion
    seqs_cpu = seqs.cpu()
    for i in range(batch_size):
        text = tokenizer.sequence_to_text(seqs_cpu[i])
        result_strings.append(text)

    return result_strings


def generate_submission(load_cached_data: bool = True):
    """
    Generates the submission file for the test set using the best trained model.

    Args:
        load_cached_data (bool): Whether to load the tokenizer vocabulary from cache.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"Generating submission on device: {device}")

    # 1. Load Tokenizer
    tokenizer = Tokenizer()
    tokenizer.build_vocab(load_cached_data=load_cached_data)

    # 2. Setup Test Dataset and Loader
    # Using 'test' transforms (Resize + Normalize)
    test_transform = get_transforms("test")

    # Ensure test metadata exists
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_METADATA}")

    test_dataset = InChiDataset(
        Config.TEST_METADATA, tokenizer, transform=test_transform, is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    vocab_size = tokenizer.get_vocab_size()
    model = ResNetTCN(vocab_size=vocab_size)
    model = model.to(device)

    checkpoint_path = Config.CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle case where checkpoint saves 'state_dict' key or just state dict
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using initialized weights (predictions will be random)."
        )

    model.eval()

    # 4. Inference Loop
    results = []

    print(f"Starting inference on {len(test_dataset)} images...")

    with torch.no_grad():
        for i, (images, image_ids) in enumerate(test_loader):
            images = images.to(device)

            # Predict batch
            predicted_inchis = greedy_decode(
                model, images, tokenizer, max_len=Config.MAX_LEN, device=device
            )

            # Store results
            for img_id, inchi in zip(image_ids, predicted_inchis):
                results.append({"image_id": img_id, "InChI": inchi})

            if (i + 1) % 100 == 0:
                print(f"Processed batch {i + 1}/{len(test_loader)}")

    # 5. Save Submission
    df = pd.DataFrame(results)

    # Sort by image_id just in case, though usually order is preserved or irrelevant
    # df = df.sort_values("image_id")

    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(df)}")
