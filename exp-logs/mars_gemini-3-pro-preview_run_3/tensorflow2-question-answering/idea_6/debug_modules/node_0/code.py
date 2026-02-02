import os
import torch
import pandas as pd
import logging
import shutil

# Import from the provided library files
from library.config import PathConfig, ModelConfig, TrainingConfig
from library.utils import set_seed, parse_html, normalize_answer, f1_score
from library.data_loader import get_dataloaders
from library.models import DualEncoderRanker, SimilarityProjectionReader
from library.train import train_ranker_epoch, train_reader_epoch
from library.evaluate import predict_submission


def run_demonstration():
    print("Starting Demonstration of QA System Components")

    # 1. Configuration Override for Speed
    # We modify the global configuration to run a fast, minimal version of the pipeline
    print("\n[1] Configuring for fast execution...")
    TrainingConfig.SUBSET_SIZE = 50  # Only use 50 examples
    TrainingConfig.BATCH_SIZE = 4  # Small batch size
    TrainingConfig.EPOCHS = 1  # Only 1 epoch
    TrainingConfig.NUM_WORKERS = (
        0  # Disable multiprocessing for simple script execution
    )

    # Redirect working directory to a demo folder to avoid conflicts
    PathConfig.WORKING_DIR = "./working/demo_execution"
    PathConfig.ensure_dirs()

    # Update derived paths in PathConfig based on the new WORKING_DIR
    PathConfig.RANKER_TRAIN_DATA = os.path.join(
        PathConfig.WORKING_DIR, "ranker_train_data.parquet"
    )
    PathConfig.RANKER_VAL_DATA = os.path.join(
        PathConfig.WORKING_DIR, "ranker_val_data.parquet"
    )
    PathConfig.READER_TRAIN_DATA = os.path.join(
        PathConfig.WORKING_DIR, "reader_train_data.parquet"
    )
    PathConfig.READER_VAL_DATA = os.path.join(
        PathConfig.WORKING_DIR, "reader_val_data.parquet"
    )
    PathConfig.RANKER_MODEL_PATH = os.path.join(
        PathConfig.WORKING_DIR, "ranker_best.pth"
    )
    PathConfig.READER_MODEL_PATH = os.path.join(
        PathConfig.WORKING_DIR, "reader_best.pth"
    )
    PathConfig.SUBMISSION_FILE = os.path.join(
        PathConfig.WORKING_DIR, "submission", "submission.csv"
    )
    os.makedirs(os.path.dirname(PathConfig.SUBMISSION_FILE), exist_ok=True)

    device = torch.device(TrainingConfig.DEVICE)
    print(f"    Device: {device}")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions...")

    # Test HTML Parsing
    raw_html = "<P> This is a <B> paragraph </B> . </P> <H1> Title </H1>"
    parsed = parse_html(raw_html)
    assert len(parsed) == 2, "Should parse into 2 candidates"
    assert parsed[0]["text"] == "This is a paragraph .", "Text cleaning failed"
    assert parsed[1]["text"] == "Title", "Header parsing failed"
    print("    parse_html: OK")

    # Test Normalization
    raw_text = "The United States of America."
    norm_text = normalize_answer(raw_text)
    assert norm_text == "united states of america", f"Normalization failed: {norm_text}"
    print("    normalize_answer: OK")

    # Test Metric
    f1 = f1_score("united states", "the united states")
    assert f1 > 0.8, "F1 score calculation seems incorrect"
    print("    f1_score: OK")

    # 3. Data Loading
    print("\n[3] Initializing DataLoaders (Processing & Caching)...")
    # This will process the first 50 examples from the metadata and cache them
    # We set load_cached_data=False to force processing for this demo
    ranker_train_dl, ranker_val_dl, reader_train_dl, reader_val_dl = get_dataloaders(
        load_cached_data=False
    )

    print(f"    Ranker Train Batches: {len(ranker_train_dl)}")
    print(f"    Reader Train Batches: {len(reader_train_dl)}")

    assert len(ranker_train_dl) > 0, "Ranker train loader is empty"
    assert len(reader_train_dl) > 0, "Reader train loader is empty"

    # 4. Model Instantiation & Forward Pass Verification
    print("\n[4] Instantiating Models...")

    # Ranker
    ranker_model = DualEncoderRanker().to(device)
    ranker_batch = next(iter(ranker_train_dl))

    # Move batch to device
    q_ids = ranker_batch["q_input_ids"].to(device)
    q_mask = ranker_batch["q_attention_mask"].to(device)

    # Forward pass
    q_emb = ranker_model(q_ids, q_mask)
    assert q_emb.shape == (q_ids.size(0), 384), "Ranker output shape mismatch"
    print("    Ranker Model Forward Pass: OK")

    # Reader
    reader_model = SimilarityProjectionReader().to(device)
    reader_batch = next(iter(reader_train_dl))

    input_ids = reader_batch["input_ids"].to(device)
    attn_mask = reader_batch["attention_mask"].to(device)
    token_type = reader_batch["token_type_ids"].to(device)

    # Forward pass
    start_logits, end_logits = reader_model(input_ids, attn_mask, token_type)
    assert start_logits.shape == input_ids.shape, "Reader start logits shape mismatch"
    assert end_logits.shape == input_ids.shape, "Reader end logits shape mismatch"
    print("    Reader Model Forward Pass: OK")

    # 5. Training Loop Simulation
    print("\n[5] Running Training Simulation (1 Epoch)...")

    # Ranker Training
    ranker_opt = torch.optim.AdamW(ranker_model.parameters(), lr=1e-5)
    ranker_crit = torch.nn.TripletMarginLoss()

    r_loss, r_acc = train_ranker_epoch(
        ranker_model, ranker_train_dl, ranker_opt, device, ranker_crit
    )
    print(f"    Ranker Train Loss: {r_loss:.4f}, Acc: {r_acc:.4f}")

    # Save Ranker
    torch.save(ranker_model.state_dict(), PathConfig.RANKER_MODEL_PATH)

    # Reader Training
    reader_opt = torch.optim.AdamW(reader_model.parameters(), lr=1e-5)
    reader_crit = torch.nn.CrossEntropyLoss()

    rd_loss = train_reader_epoch(
        reader_model, reader_train_dl, reader_opt, device, reader_crit
    )
    print(f"    Reader Train Loss: {rd_loss:.4f}")

    # Save Reader
    torch.save(reader_model.state_dict(), PathConfig.READER_MODEL_PATH)
    print("    Models saved successfully.")

    # 6. Evaluation / Inference
    print("\n[6] Running Inference on Test Set...")
    # Run prediction on a tiny subset of test data
    predict_submission(subset_size=10)

    # Verify submission file
    if os.path.exists(PathConfig.SUBMISSION_FILE):
        df = pd.read_csv(PathConfig.SUBMISSION_FILE)
        print(f"    Submission file created with {len(df)} rows.")
        print("    First few rows:")
        print(df.head())

        # Basic checks
        assert len(df) == 20, "Expected 20 rows (10 examples * 2 types)"
        assert "example_id" in df.columns
        assert "PredictionString" in df.columns
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration Completed Successfully.")


if __name__ == "__main__":
    run_demonstration()
