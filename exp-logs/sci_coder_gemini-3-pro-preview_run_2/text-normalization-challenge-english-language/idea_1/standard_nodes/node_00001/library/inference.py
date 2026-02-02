import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from library.config import Config
from library.vocabulary import get_vocab
from library.dataset import TextNormalizationDataset
from library.model import Encoder, Decoder, Attention, Seq2Seq


def generate_submission(load_cached_data=True, debug=False):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        load_cached_data (bool): Whether to load pre-processed dataset from cache.
        debug (bool): Whether to run in debug mode (subset of data).
    """
    # 1. Setup
    Config.set_seed()
    device = Config.get_device()
    print(f"Inference Device: {device}")

    # 2. Load Vocabulary
    vocab = get_vocab(load_cached_data=load_cached_data)
    input_dim = len(vocab)
    output_dim = len(vocab)

    # 3. Load Test Data
    print("Loading test dataset...")
    test_dataset = TextNormalizationDataset(
        split="test", vocab=vocab, load_cached_data=load_cached_data, debug=debug
    )

    # Use a larger batch size for inference since we don't store gradients
    inference_batch_size = Config.BATCH_SIZE

    test_loader = DataLoader(
        test_dataset,
        batch_size=inference_batch_size,
        shuffle=False,
        collate_fn=TextNormalizationDataset.collate_fn,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 4. Initialize Model Architecture
    print("Initializing model architecture...")
    attn = Attention(Config.HIDDEN_DIM, Config.HIDDEN_DIM)

    enc = Encoder(
        input_dim=input_dim,
        emb_dim=Config.ENC_EMB_DIM,
        enc_hid_dim=Config.HIDDEN_DIM,
        dec_hid_dim=Config.HIDDEN_DIM,
        dropout=Config.ENC_DROPOUT,
    )

    dec = Decoder(
        output_dim=output_dim,
        emb_dim=Config.DEC_EMB_DIM,
        enc_hid_dim=Config.HIDDEN_DIM,
        dec_hid_dim=Config.HIDDEN_DIM,
        dropout=Config.DEC_DROPOUT,
        attention=attn,
    )

    model = Seq2Seq(enc, dec, device).to(device)

    # 5. Load Model Weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading model weights from {Config.MODEL_SAVE_PATH}")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Train the model first."
        )

    # 6. Inference Loop
    print("Starting inference...")
    model.eval()

    results_id = []
    results_text = []

    with torch.no_grad():
        for batch in test_loader:
            src = batch["src"].to(device)
            src_len = batch["src_len"].to("cpu")
            ids = batch["id"]

            # Predict using greedy decoding
            # Returns: [batch_size, max_len]
            predictions = model.predict(src, src_len, max_len=Config.MAX_SEQ_LEN)

            # Decode predictions
            for i in range(predictions.shape[0]):
                # Convert indices to string
                pred_indices = predictions[i].cpu().tolist()
                decoded_text = vocab.decode(pred_indices, remove_special=True)

                results_id.append(ids[i])
                results_text.append(decoded_text)

    # 7. Create Submission DataFrame
    print("Creating submission file...")
    submission_df = pd.DataFrame({"id": results_id, "after": results_text})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # Save to CSV
    # The submission format requires quoting for text fields usually,
    # pandas handles this automatically.
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)

    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(f"Total predictions generated: {len(submission_df)}")
