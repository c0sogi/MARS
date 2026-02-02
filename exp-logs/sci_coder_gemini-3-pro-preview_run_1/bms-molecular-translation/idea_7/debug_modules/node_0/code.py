import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.utils import parse_inchi_attributes, compute_attribute_stats
from library.tokenizer import Tokenizer
from library.data import InChiDataset, get_transforms, get_dataloaders
from library.modules import AdaLN, AdaLNDecoderLayer
from library.model import AMViT
from library.train import Trainer
from library.inference import Predictor


def run_demonstration():
    print("=== Starting Demonstration of InChI Prediction Library ===\n")

    # 1. Setup and Configuration Overrides for Speed
    print("--- 1. Configuration Setup ---")
    seed_everything(Config.SEED)

    # Override Config for fast execution
    Config.DEBUG = True
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.TOKENIZER_PATH = os.path.join(Config.WORKING_DIR, "demo_tokenizer.json")
    Config.ATTR_STATS_PATH = os.path.join(Config.WORKING_DIR, "demo_attr_stats.npy")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    print("Configuration updated for speed and demo paths.\n")

    # 2. Verify Utility Functions
    print("--- 2. Verifying Utils ---")
    # Test InChI parsing
    test_inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
    # Expected: C=2, H=6, O=1, N=0, S=0, Halogen=0, Length=len(str)
    # Order: ["C", "H", "O", "N", "S", "Halogen", "Length"]
    attrs = parse_inchi_attributes(test_inchi)
    print(f"Parsed attributes for {test_inchi}: {attrs}")

    assert attrs[0] == 2.0, "Carbon count mismatch"
    assert attrs[1] == 6.0, "Hydrogen count mismatch"
    assert attrs[2] == 1.0, "Oxygen count mismatch"
    assert attrs[-1] == len(test_inchi), "Length mismatch"
    print("parse_inchi_attributes logic verified.")

    # Test Attribute Stats Computation on a dummy dataframe
    dummy_data = {
        "image_id": ["id1", "id2"],
        "InChI": ["InChI=1S/CH4/h1H4", "InChI=1S/H2O/h1H2"],
        "file_path": ["dummy/path1.png", "dummy/path2.png"],
    }
    dummy_df = pd.DataFrame(dummy_data)
    stats = compute_attribute_stats(train_df=dummy_df, load_cached_data=False)
    assert stats.shape == (2, Config.NUM_ATTRIBUTES), "Stats shape mismatch"
    print("compute_attribute_stats logic verified.\n")

    # 3. Verify Tokenizer
    print("--- 3. Verifying Tokenizer ---")
    tokenizer = Tokenizer()
    texts = ["InChI=1S/C", "InChI=1S/H"]
    tokenizer.fit_on_texts(texts=texts, load_cached_data=False)

    encoded = tokenizer.encode("InChI=1S/C")
    print(f"Encoded 'InChI=1S/C': {encoded[:15]}...")
    assert len(encoded) == Config.MAX_LEN, "Encoded sequence length mismatch"
    assert encoded[0] == tokenizer.token2id[tokenizer.sos_token], "Start token mismatch"

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    # Note: decode skips SOS and stops at EOS, so we expect the content string
    assert decoded == "InChI=1S/C", "Decode mismatch"
    print("Tokenizer encoding/decoding verified.\n")

    # 4. Verify Data Loading
    print("--- 4. Verifying Data Loading ---")
    # We use get_dataloaders with debug=True which loads a small subset of real metadata
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(debug=True)

    batch = next(iter(train_loader))
    images = batch["image"]
    text_seq = batch["text_seq"]
    attributes = batch["attributes"]

    print(f"Batch keys: {batch.keys()}")
    print(f"Image batch shape: {images.shape}")
    print(f"Text sequence shape: {text_seq.shape}")
    print(f"Attributes shape: {attributes.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    ), "Image tensor shape incorrect"
    assert text_seq.shape == (
        Config.BATCH_SIZE,
        Config.MAX_LEN,
    ), "Text sequence shape incorrect"
    assert attributes.shape == (
        Config.BATCH_SIZE,
        Config.NUM_ATTRIBUTES,
    ), "Attributes shape incorrect"
    print("Data loading verified.\n")

    # 5. Verify Model Components
    print("--- 5. Verifying Model Components ---")
    device = torch.device("cpu")  # Use CPU for simple component check

    # AdaLN
    embed_dim = 32
    cond_dim = 7
    adaln = AdaLN(embed_dim, cond_dim).to(device)
    dummy_x = torch.randn(2, 10, embed_dim).to(device)
    dummy_cond = torch.randn(2, cond_dim).to(device)
    out = adaln(dummy_x, dummy_cond)
    assert out.shape == dummy_x.shape, "AdaLN output shape mismatch"
    print("AdaLN verified.")

    # Decoder Layer
    decoder_layer = AdaLNDecoderLayer(
        embed_dim, cond_dim, num_heads=4, ff_dim=64, encoder_dim=64
    ).to(device)
    dummy_enc_out = torch.randn(2, 20, 64).to(device)  # B, L_enc, Enc_dim
    out = decoder_layer(dummy_x, dummy_enc_out, dummy_cond)
    assert out.shape == dummy_x.shape, "Decoder Layer output shape mismatch"
    print("AdaLNDecoderLayer verified.")

    # Full AMViT Model
    vocab_size = len(tokenizer)
    model = AMViT(vocab_size=vocab_size, pad_idx=0).to(device)
    # Use the batch from data loader (move to cpu)
    img_batch = images.to(device)
    seq_batch = text_seq[:, :-1].to(device)  # Input for teacher forcing

    logits, pred_attrs = model(img_batch, seq_batch)
    print(f"Model Logits Shape: {logits.shape}")
    print(f"Predicted Attributes Shape: {pred_attrs.shape}")

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.MAX_LEN - 1,
        vocab_size,
    ), "Logits shape mismatch"
    assert pred_attrs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_ATTRIBUTES,
    ), "Predicted attributes shape mismatch"
    print("AMViT Model forward pass verified.\n")

    # 6. Verify Training Loop
    print("--- 6. Verifying Trainer (Fit) ---")
    # Initialize Trainer with debug=True
    trainer = Trainer(debug=True)
    # Ensure the trainer uses our modified config
    trainer.model.to(trainer.device)

    # Run a short training cycle
    print("Running trainer.fit() with 1 epoch on subset...")
    trainer.fit()

    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not saved."
    print("Trainer fit execution verified.\n")

    # 7. Verify Inference
    print("--- 7. Verifying Inference (Predictor) ---")
    predictor = Predictor(model_path=Config.MODEL_PATH)

    # Run prediction on the debug test loader
    print("Running predictor.generate_submission()...")
    predictor.generate_submission(test_loader, submission_path=Config.SUBMISSION_FILE)

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    assert len(df_sub) > 0, "Submission file is empty."
    assert (
        "image_id" in df_sub.columns and "InChI" in df_sub.columns
    ), "Submission columns missing."
    print("Inference execution verified.\n")

    print("=== Demonstration Complete: All components verified successfully ===")


if __name__ == "__main__":
    run_demonstration()
