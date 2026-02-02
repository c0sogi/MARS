import os
import time
import re
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.tokenizer import HybridTokenizer
from library.hfbb import HierarchicalBackoff
from library.transformer_model import TransformerTrainer, NormalizationDataset
from library.data_factory import create_dataloaders, _add_context


def train_model():
    """
    Orchestrates the training of the Text Normalization model.
    1. Prepares Tokenizer and HFBB (Tier 1).
    2. Loads data with Soft-Residual weights.
    3. Trains the Transformer (Tier 2) using weighted loss.
    4. Generates submission on the test set.
    """
    set_seed(Config.SEED)

    # ==========================================
    # 1. Component Initialization
    # ==========================================
    print("Initializing and training Tokenizer...")
    tokenizer = HybridTokenizer()
    tokenizer.train(load_cached_data=True)

    print("Initializing and fitting HFBB (Tier 1)...")
    hfbb = HierarchicalBackoff()
    hfbb.fit(load_cached_data=True)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Creating DataLoaders...")
    # create_dataloaders handles caching and soft-residual weight computation via HFBB
    train_loader, val_loader = create_dataloaders(
        tokenizer, hfbb, load_cached_data=True
    )

    # ==========================================
    # 3. Model Setup
    # ==========================================
    print("Initializing Transformer Trainer...")
    # We use TransformerTrainer as a container for model, optimizer, and criterion
    trainer = TransformerTrainer(tokenizer)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print(f"Starting training on {Config.DEVICE}...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()
        trainer.model.train()
        total_train_loss = 0

        for batch in train_loader:
            src = batch["src"].to(trainer.device)
            tgt = batch["tgt"].to(trainer.device)
            weights = batch["weight"].to(trainer.device)

            # Prepare inputs and targets for Teacher Forcing
            tgt_input = tgt[:, :-1]  # <SOS> ... <last>
            tgt_output = tgt[:, 1:]  # ... <last> <EOS>

            # Create Masks
            src_pad_mask = src == tokenizer.char2id[tokenizer.PAD_TOKEN]
            tgt_pad_mask = tgt_input == tokenizer.pad_id
            tgt_mask = trainer.model.generate_square_subsequent_mask(
                tgt_input.size(1)
            ).to(trainer.device)

            trainer.optimizer.zero_grad()

            # Forward Pass
            logits = trainer.model(
                src,
                tgt_input,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                tgt_mask=tgt_mask,
            )

            # Loss Calculation (Weighted)
            # Reshape logits: (Batch * SeqLen, VocabSize)
            # Reshape target: (Batch * SeqLen)
            loss_per_token = trainer.criterion(
                logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
            )

            # Expand weights to match sequence length
            # weights: (Batch) -> (Batch, SeqLen) -> (Batch * SeqLen)
            weights_expanded = weights.unsqueeze(1).expand_as(tgt_output).reshape(-1)
            weighted_loss = loss_per_token * weights_expanded

            # Calculate mean loss over non-pad tokens
            non_pad_mask = tgt_output.reshape(-1) != tokenizer.pad_id
            if non_pad_mask.sum() > 0:
                loss = weighted_loss[non_pad_mask].mean()
            else:
                loss = weighted_loss.mean()

            # Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), Config.MAX_GRAD_NORM
            )
            trainer.optimizer.step()

            total_train_loss += loss.item()

        # Calculate Epoch Metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = trainer.evaluate(val_loader)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {elapsed:.0f}s | "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        # Checkpointing & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(trainer.model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  New best model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # Load best model for inference
    if os.path.exists(Config.BEST_MODEL_PATH):
        print("Loading best model for submission generation...")
        trainer.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=trainer.device)
        )

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    generate_submission(trainer, tokenizer, hfbb)


def generate_submission(trainer, tokenizer, hfbb):
    """
    Generates predictions for the test set using the Hybrid Cascade strategy.
    """
    print("Generating submission...")
    set_seed(Config.SEED)

    # 1. Load Test Data
    # We use the metadata test file which contains sentence_id, token_id, before
    df_test = pd.read_csv(Config.TEST_DATA)
    df_test["before"] = df_test["before"].fillna("").astype(str)

    # 2. Add Context
    # We use the internal helper from data_factory to ensure consistency
    print("Adding context to test data...")
    df_test = _add_context(df_test)

    # 3. Hybrid Routing Logic
    print("Applying HFBB and Routing Logic...")

    # Pre-calculate IDs for submission
    ids = df_test.apply(
        lambda x: f"{x['sentence_id']}_{x['token_id']}", axis=1
    ).tolist()

    # Containers
    predictions = [None] * len(df_test)
    nn_indices = []  # Indices of rows requiring Neural Network prediction

    # Regex for Semiotic check (Digits or Latin)
    semiotic_pattern = re.compile(r"\d|[a-zA-Z]")

    # Iterate through test data
    # (Using lists for speed over pandas iteration)
    befores = df_test["before"].tolist()
    prevs = df_test["prev"].tolist()
    nexts = df_test["next"].tolist()

    for idx, (before, prev, nxt) in enumerate(zip(befores, prevs, nexts)):
        # Step A: Tier 1 (HFBB) Query
        pred, conf, level = hfbb.query(before, prev, nxt)

        # Logic: Accept HFBB if:
        # 1. It's a context-specific match (trigram/bigram) -> High Specificity
        # 2. It's a Unigram match with High Confidence -> High Stability
        if pred is not None and (
            level != "unigram" or conf > Config.HFBB_CONFIDENCE_THRESHOLD
        ):
            predictions[idx] = pred
        else:
            # Step B: Tier 2 Candidate Check
            # If the token contains digits or latin characters, it's "Semiotic" and needs normalization
            if semiotic_pattern.search(before):
                nn_indices.append(idx)
            else:
                # Step C: Identity Fallback
                # Plain words usually don't change
                predictions[idx] = before

    print(
        f"Total tokens: {len(df_test)}. "
        f"Handled by HFBB/Identity: {len(df_test) - len(nn_indices)}. "
        f"Routed to Transformer: {len(nn_indices)}"
    )

    # 4. Run Transformer on Residuals
    if nn_indices:
        # Create subset dataframe
        df_nn = df_test.iloc[nn_indices].copy()

        # Create Dataset and Loader
        # is_train=False ensures we only get 'src'
        nn_dataset = NormalizationDataset(df_nn, tokenizer, is_train=False)
        nn_loader = DataLoader(
            nn_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        trainer.model.eval()
        nn_preds = []

        with torch.no_grad():
            for batch in nn_loader:
                src = batch["src"].to(trainer.device)
                # Greedy decoding
                tgt_indices = trainer.predict(src)

                # Decode BPE to String
                decoded_batch = [tokenizer.decode(t) for t in tgt_indices]
                nn_preds.extend(decoded_batch)

        # Assign predictions back to the main list
        for i, pred in enumerate(nn_preds):
            original_idx = nn_indices[i]
            predictions[original_idx] = pred

    # 5. Save Submission
    sub_df = pd.DataFrame({"id": ids, "after": predictions})

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generation complete.")
