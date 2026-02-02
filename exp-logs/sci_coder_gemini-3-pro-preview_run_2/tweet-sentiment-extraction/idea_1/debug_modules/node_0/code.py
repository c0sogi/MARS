import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library
from library import config
from library import utils
from library import data_loader
from library import model
from library import engine


def run_demo():
    print("=== Starting Library Demo ===")

    # 1. Verify Utilities
    print("\n[1] Verifying Utilities...")
    s1 = "this is a test"
    s2 = "this is a test"
    s3 = "this is not a test"

    score_perfect = utils.jaccard(s1, s2)
    score_partial = utils.jaccard(s1, s3)

    print(f"   Jaccard '{s1}' vs '{s2}': {score_perfect}")
    print(f"   Jaccard '{s1}' vs '{s3}': {score_partial:.4f}")

    assert score_perfect == 1.0, "Jaccard score for identical strings should be 1.0"
    assert (
        0.0 < score_partial < 1.0
    ), "Jaccard score for partial match should be between 0 and 1"
    print("   -> Utilities verified.")

    # 2. Data Loading and Processing
    print("\n[2] Verifying Data Loading & Processing...")

    # Load a small subset of the training data for demonstration speed
    df_full = pd.read_csv(config.TRAIN_PATH)
    # Filter out NaNs just in case, though metadata is clean
    df_full.dropna(subset=["text", "selected_text", "sentiment"], inplace=True)

    # Take a small subset (e.g., 50 samples)
    subset_size = 50
    df_subset = df_full.head(subset_size).copy().reset_index(drop=True)
    print(f"   Created subset of {len(df_subset)} samples.")

    # Initialize and Build Vocabulary
    vocab = data_loader.Vocabulary()
    vocab.build(df_subset["text"].tolist())
    print(f"   Vocabulary size: {len(vocab)}")

    # Process Data (Tokenization, Padding, Label Generation)
    # We use the config.MAX_LEN and SENTIMENT_MAP
    processed_data = data_loader.process_data(
        df_subset, vocab, config.MAX_LEN, config.SENTIMENT_MAP, is_test=False
    )

    # Verify processed data shapes
    assert len(processed_data["input_ids"]) == subset_size
    assert processed_data["input_ids"].shape[1] == config.MAX_LEN
    assert processed_data["start_ids"].shape[0] == subset_size
    print("   Processed data shapes are correct.")

    # Create Dataset and DataLoader
    dataset = data_loader.TweetDataset(df_subset, processed_data)
    batch_size = 8
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Fetch one batch to verify structure
    batch = next(iter(loader))
    print("   Batch keys:", batch.keys())
    assert "input_ids" in batch
    assert "sentiment_id" in batch
    assert "start_idx" in batch
    assert batch["input_ids"].shape == (batch_size, config.MAX_LEN)
    print("   -> Data Loading verified.")

    # 3. Model Initialization and Forward Pass
    print("\n[3] Verifying Model Architecture...")

    device = config.DEVICE
    net = model.BiGRUPointerNetwork(vocab_size=len(vocab)).to(device)

    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    sentiment_ids = batch["sentiment_id"].to(device)

    # Forward pass
    start_logits, end_logits = net(input_ids, sentiment_ids, attention_mask)

    print(f"   Logits Shape: {start_logits.shape}")

    # Verify output shape: (batch_size, seq_len)
    assert start_logits.shape == (batch_size, config.MAX_LEN)
    assert end_logits.shape == (batch_size, config.MAX_LEN)
    print("   -> Model Forward Pass verified.")

    # 4. Engine Logic (Loss, Decoding, Training Step)
    print("\n[4] Verifying Engine Logic...")

    # Loss Calculation
    start_targets = batch["start_idx"].to(device)
    end_targets = batch["end_idx"].to(device)

    loss = engine.loss_fn(start_logits, end_logits, start_targets, end_targets)
    print(f"   Calculated Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"

    # Decoding
    start_preds, end_preds = engine.decode_span(
        start_logits, end_logits, attention_mask
    )
    print(f"   Decoded Spans (first 3): {list(zip(start_preds, end_preds))[:3]}")

    assert len(start_preds) == batch_size
    # Check that start <= end for all predictions
    for s, e in zip(start_preds, end_preds):
        assert s <= e, f"Start index {s} must be <= End index {e}"

    # Optimization Step
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check if weights updated (simple check on gradients)
    assert net.start_head.weight.grad is not None, "Gradients should be computed"
    print("   -> Engine Logic verified.")

    # 5. Integration: Mini-Training Loop
    print("\n[5] Running Mini-Training Loop (Integration Test)...")

    # We use the small loader created above
    # We will run for 2 epochs to demonstrate the loop without taking too much time
    demo_epochs = 2

    for epoch in range(demo_epochs):
        train_loss = engine.train_fn(loader, net, optimizer, device)
        # We use the same loader for eval just to demonstrate the function call
        val_loss, val_jaccard = engine.eval_fn(loader, net, device)

        print(
            f"   Epoch {epoch+1}/{demo_epochs} | Train Loss: {train_loss:.4f} | Val Jaccard: {val_jaccard:.4f}"
        )

    print("   -> Training Loop completed.")

    # 6. Inference Demonstration
    print("\n[6] Inference Demonstration...")

    # Create a dummy test set
    test_df = df_subset.head(5).copy()
    # Drop selected_text to simulate test data
    test_df_input = test_df.drop(columns=["selected_text"])

    # Process test data
    test_processed = data_loader.process_data(
        test_df_input, vocab, config.MAX_LEN, config.SENTIMENT_MAP, is_test=True
    )
    test_dataset = data_loader.TweetDataset(test_df_input, test_processed)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    # Run prediction logic manually (similar to engine.predict_fn but returning results)
    net.eval()
    results = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_ids = batch["sentiment_id"].to(device)
            texts = batch["text"]
            text_ids = batch["textID"]

            s_logits, e_logits = net(input_ids, sentiment_ids, attention_mask)
            s_preds, e_preds = engine.decode_span(s_logits, e_logits, attention_mask)

            for k in range(len(texts)):
                text_tokens = texts[k].split()
                i, j = s_preds[k], e_preds[k]
                pred_text = " ".join(text_tokens[i : j + 1])
                results.append({"textID": text_ids[k], "selected_text": pred_text})

    # Display results
    print("   Sample Predictions:")
    for res in results:
        print(f"     ID: {res['textID']} -> Pred: '{res['selected_text']}'")

    # Verify submission file generation
    sub_df = pd.DataFrame(results)
    output_path = "./working/demo_submission.csv"
    sub_df.to_csv(output_path, index=False)
    assert os.path.exists(output_path)
    print(f"   -> Inference verified. Output saved to {output_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    config.seed_everything(config.SEED)
    run_demo()
