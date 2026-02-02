import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import csv

from library.config import Config
from library.utils import setup_logger, save_npy, load_npy, ensure_dir
from library.models import PriorAugmentedBiLSTMTagger, TransformerSeq2Seq
from library.data_loader import TaggerDataset
from library.features import RegexFeatureExtractor

logger = setup_logger("engine")


def get_class_weights(vocab_manager, load_cached_data=True):
    """
    Computes or loads class weights for the Tagger loss.
    Weights = sqrt(Total_Samples / Class_Count).
    """
    weights_path = os.path.join(Config.WORK_DIR, "class_weights.npy")

    if load_cached_data and os.path.exists(weights_path):
        logger.info(f"Loading class weights from {weights_path}")
        weights = load_npy(weights_path)
        return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)

    logger.info("Computing class weights from training data...")
    # Read training data class column
    df = pd.read_csv(Config.TRAIN_FILE, usecols=["class"])
    class_counts = df["class"].value_counts().to_dict()

    class_vocab = vocab_manager.get_class_vocab()
    num_classes = len(class_vocab)
    total_samples = len(df)

    weights = np.zeros(num_classes, dtype=np.float32)

    for cls_name, idx in class_vocab.token2idx.items():
        count = class_counts.get(cls_name, 0)
        if count > 0:
            # Square-root smoothing
            weights[idx] = np.sqrt(total_samples / count)
        else:
            # Fallback for classes not in train set (unlikely)
            weights[idx] = 1.0

    # Normalize weights so mean is 1.0 (optional but good for stability)
    weights = weights / weights.mean()

    save_npy(weights, weights_path)
    logger.info(f"Class weights computed and saved to {weights_path}")

    return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)


def train_tagger(dataloaders, vocab_manager, prior_manager, load_cached_data=True):
    """
    Trains the Prior-Augmented Bi-LSTM Tagger.
    """
    logger.info("Initializing Tagger Training...")

    device = Config.DEVICE
    regex_extractor = RegexFeatureExtractor()
    regex_dim = regex_extractor.get_feature_dim()
    num_classes = len(vocab_manager.get_class_vocab())

    # Model
    model = PriorAugmentedBiLSTMTagger(vocab_manager, regex_dim, num_classes).to(device)

    # Loss with Class Weights
    class_weights = get_class_weights(vocab_manager, load_cached_data)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    # Optimizer & Scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    train_loader = dataloaders["tagger"]["train"]
    val_loader = dataloaders["tagger"]["val"]

    logger.info(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]", leave=False
        ):
            word_ids = batch["word_ids"].to(device)
            bpe_ids = batch["bpe_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            regex_feats = batch["regex_feats"].to(device)
            prior_feats = batch["prior_feats"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            logits = model(word_ids, bpe_ids, char_ids, regex_feats, prior_feats)

            # Flatten for loss: (B*S, NumClasses) vs (B*S)
            loss = criterion(logits.view(-1, num_classes), targets.view(-1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=2)
                mask = targets != -1
                correct = (preds == targets) & mask
                train_correct += correct.sum().item()
                train_total += mask.sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]", leave=False
            ):
                word_ids = batch["word_ids"].to(device)
                bpe_ids = batch["bpe_ids"].to(device)
                char_ids = batch["char_ids"].to(device)
                regex_feats = batch["regex_feats"].to(device)
                prior_feats = batch["prior_feats"].to(device)
                targets = batch["targets"].to(device)

                logits = model(word_ids, bpe_ids, char_ids, regex_feats, prior_feats)
                loss = criterion(logits.view(-1, num_classes), targets.view(-1))

                val_loss += loss.item()

                preds = torch.argmax(logits, dim=2)
                mask = targets != -1
                correct = (preds == targets) & mask
                val_correct += correct.sum().item()
                val_total += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        logger.info(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Train Acc={train_acc:.6f}, "
            f"Val Loss={avg_val_loss:.6f}, Val Acc={val_acc:.6f}"
        )

        # Scheduler Step
        scheduler.step(avg_val_loss)

        # Checkpoint & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.TAGGER_MODEL_PATH)
            logger.info(f"New best model saved to {Config.TAGGER_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                logger.info("Early stopping triggered.")
                break

    # Load best model before returning
    model.load_state_dict(torch.load(Config.TAGGER_MODEL_PATH, map_location=device))
    return model


def train_seq2seq(dataloaders, vocab_manager):
    """
    Trains the Transformer Seq2Seq Fallback Model.
    """
    logger.info("Initializing Seq2Seq Training...")

    device = Config.DEVICE
    num_classes = len(vocab_manager.get_class_vocab())
    pad_idx = vocab_manager.get_char_vocab()["<pad>"]

    model = TransformerSeq2Seq(vocab_manager, num_classes).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    patience_counter = 0

    train_loader = dataloaders["seq2seq"]["train"]
    val_loader = dataloaders["seq2seq"]["val"]

    logger.info(f"Starting Seq2Seq training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]", leave=False
        ):
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            class_ids = batch["class_ids"].to(device)

            # Target Input: <sos> ... token_n
            tgt_input = tgt_ids[:, :-1]
            # Target Output: token_1 ... <eos>
            tgt_output = tgt_ids[:, 1:]

            optimizer.zero_grad()

            # Forward
            logits = model(src_ids, tgt_input, class_ids)

            # Reshape for loss: (B * Tgt_Len, Vocab)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]", leave=False
            ):
                src_ids = batch["src_ids"].to(device)
                tgt_ids = batch["tgt_ids"].to(device)
                class_ids = batch["class_ids"].to(device)

                tgt_input = tgt_ids[:, :-1]
                tgt_output = tgt_ids[:, 1:]

                logits = model(src_ids, tgt_input, class_ids)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        logger.info(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.SEQ2SEQ_MODEL_PATH)
            logger.info(f"New best Seq2Seq model saved to {Config.SEQ2SEQ_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                logger.info("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(Config.SEQ2SEQ_MODEL_PATH, map_location=device))
    return model


def generate_submission(tagger_model, seq2seq_model, vocab_manager, prior_manager, kb):
    """
    Generates predictions for the test set and saves the submission file.
    """
    logger.info("Generating submission...")
    device = Config.DEVICE

    # 1. Create Test Loader
    test_ds = TaggerDataset(
        Config.TEST_FILE,
        vocab_manager,
        prior_manager,
        mode="test",
        load_cached_data=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=TaggerDataset.collate_fn,
    )

    tagger_model.eval()
    seq2seq_model.eval()

    class_vocab = vocab_manager.get_class_vocab()
    char_vocab = vocab_manager.get_char_vocab()

    # Get special indices
    plain_class_idx = class_vocab["PLAIN"]

    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting", leave=True):
            word_ids = batch["word_ids"].to(device)
            bpe_ids = batch["bpe_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            regex_feats = batch["regex_feats"].to(device)
            prior_feats = batch["prior_feats"].to(device)
            # mask = batch['mask'].to(device) # Not strictly needed as we iterate by length
            batch_ids = batch["ids"]  # List of lists of strings

            # 1. Tagger Prediction
            logits = tagger_model(word_ids, bpe_ids, char_ids, regex_feats, prior_feats)
            pred_class_indices = torch.argmax(logits, dim=2)  # (B, S)

            # Iterate over batch
            batch_size = word_ids.size(0)

            # Prepare fallback batch
            fallback_indices = []  # (batch_idx, seq_idx)
            fallback_tokens = []
            fallback_class_ids = []

            # Temporary storage for this batch's predictions
            batch_preds = {}  # (batch_idx, seq_idx) -> normalized_text

            for b in range(batch_size):
                # Get actual sequence length from ids (since ids are list of lists)
                seq_len = len(batch_ids[b])

                for s in range(seq_len):
                    token_id_str = batch_ids[b][s]

                    # Get original token text (need to reverse lookup from word_ids?
                    # No, TaggerDataset doesn't return raw text.
                    # But we need raw text for KB lookup.
                    # We can reconstruct from word_ids if vocab is perfect, but OOV words are <unk>.
                    # We MUST have raw text.
                    # TaggerDataset stores it in self.data, but collate doesn't pass it.
                    # FIX: We need raw tokens.
                    # However, we can't modify TaggerDataset easily now.
                    # Workaround: TaggerDataset returns 'ids'.
                    # We can load the test csv separately to map id -> token.
                    pass

            # Since we cannot modify data_loader.py, we must load the test map.
            # This is done once outside the loop.
            pass

    # --- REVISED STRATEGY FOR RAW TEXT ACCESS ---
    # Since we need raw text for KB lookup and we can't change the dataloader,
    # we will load the test metadata into a dictionary: id -> token.
    logger.info("Loading test metadata for raw text lookup...")
    df_test = pd.read_csv(Config.TEST_FILE)
    # Ensure id column
    if "id" not in df_test.columns:
        df_test["id"] = (
            df_test["sentence_id"].astype(str) + "_" + df_test["token_id"].astype(str)
        )
    id_to_token = dict(zip(df_test["id"], df_test["before"].astype(str)))

    # Re-start loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting", leave=True):
            word_ids = batch["word_ids"].to(device)
            bpe_ids = batch["bpe_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            regex_feats = batch["regex_feats"].to(device)
            prior_feats = batch["prior_feats"].to(device)
            batch_ids = batch["ids"]

            logits = tagger_model(word_ids, bpe_ids, char_ids, regex_feats, prior_feats)
            pred_class_indices = torch.argmax(logits, dim=2).cpu().numpy()

            batch_size = word_ids.size(0)

            # Collect items needing Seq2Seq
            seq2seq_inputs = []  # (batch_idx, seq_idx, raw_token, class_idx)

            for b in range(batch_size):
                seq_len = len(batch_ids[b])
                for s in range(seq_len):
                    row_id = batch_ids[b][s]
                    raw_token = id_to_token[row_id]
                    class_idx = pred_class_indices[b, s]
                    class_name = class_vocab.lookup_token(class_idx)

                    # 1. Try KB
                    kb_result = kb.lookup(raw_token, class_name)

                    if kb_result is not None:
                        results.append((row_id, kb_result))
                    elif class_idx == plain_class_idx:
                        # PLAIN OOV -> Copy
                        results.append((row_id, raw_token))
                    else:
                        # Fallback to Seq2Seq
                        seq2seq_inputs.append((row_id, raw_token, class_idx))

            # Run Seq2Seq for this batch's fallbacks
            if seq2seq_inputs:
                # Prepare batch
                fb_src_ids = []
                fb_class_ids = []
                fb_row_ids = []

                for r_id, token, c_idx in seq2seq_inputs:
                    # Encode chars
                    c_ids = [char_vocab[c] for c in token]
                    fb_src_ids.append(torch.tensor(c_ids, dtype=torch.long))
                    fb_class_ids.append(c_idx)
                    fb_row_ids.append(r_id)

                # Pad
                fb_src_padded = torch.nn.utils.rnn.pad_sequence(
                    fb_src_ids, batch_first=True, padding_value=0
                ).to(device)
                fb_class_tensor = torch.tensor(fb_class_ids, dtype=torch.long).to(
                    device
                )

                # Generate
                generated_ids = seq2seq_model.generate(fb_src_padded, fb_class_tensor)

                # Decode
                generated_ids = generated_ids.cpu().numpy()
                for i, g_ids in enumerate(generated_ids):
                    # Stop at EOS
                    tokens = []
                    for idx in g_ids:
                        if idx == char_vocab["<eos>"]:
                            break
                        if idx not in [char_vocab["<pad>"], char_vocab["<sos>"]]:
                            tokens.append(char_vocab.lookup_token(idx))

                    norm_text = "".join(tokens)
                    results.append((fb_row_ids[i], norm_text))

    # Save Submission
    logger.info(f"Saving submission to {Config.SUBMISSION_PATH}...")
    ensure_dir(Config.SUBMISSION_PATH)

    # Sort by ID to match sample submission order if possible, though not strictly required
    # But for consistency, let's just write.
    # The sample submission has quoted strings.

    with open(Config.SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["id", "after"])
        for row_id, after_text in results:
            writer.writerow([row_id, after_text])

    logger.info("Submission generated successfully.")
