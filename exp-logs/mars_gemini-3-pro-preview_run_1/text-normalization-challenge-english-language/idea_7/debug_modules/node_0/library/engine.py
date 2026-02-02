import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import save_checkpoint, load_checkpoint, calculate_accuracy


def get_class_weights(dataset, vocab, device):
    """
    Calculates or loads class weights for the Tagger based on training data distribution.
    Uses caching to avoid re-computation.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "class_weights.npy")

    if os.path.exists(cache_path):
        print(f"Loading class weights from {cache_path}")
        weights = np.load(cache_path)
        return torch.tensor(weights, dtype=torch.float32).to(device)

    print("Calculating class weights from training data...")
    # Access the underlying dataframe from the dataset
    if hasattr(dataset, "data") and isinstance(dataset.data, pd.DataFrame):
        # Explode the list of class_ids to get individual counts
        all_classes = dataset.data["class_ids"].explode().astype(int)
        counts = all_classes.value_counts().sort_index()

        # Initialize weights with 1.0
        num_classes = len(vocab.class2id)
        weights = np.ones(num_classes, dtype=np.float32)

        total_samples = len(all_classes)

        for class_id, count in counts.items():
            if 0 <= class_id < num_classes:
                # Formula: w_c = (N / N_c) ^ alpha
                # We add a small epsilon to count to avoid division by zero if any weirdness
                w = (total_samples / (count + 1e-5)) ** Config.CLASS_WEIGHT_SMOOTHING
                weights[class_id] = w

        # Normalize weights so mean is 1.0 (keeps learning rate scale consistent)
        weights = weights / weights.mean()

        # Save to cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(cache_path, weights)
        print(
            f"Class weights calculated and saved. Max: {weights.max():.2f}, Min: {weights.min():.2f}"
        )

        return torch.tensor(weights, dtype=torch.float32).to(device)
    else:
        print(
            "Warning: Could not access dataframe for class weights. Using uniform weights."
        )
        return None


class TaggerEngine:
    """
    Engine for training and evaluating the Attention-Augmented Bi-LSTM Tagger.
    """

    def __init__(self, model, device, train_loader, val_loader, class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function with class weights and ignore_index for padding
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=1, verbose=True
        )

        self.best_acc = 0.0
        self.start_epoch = 0

    def load_checkpoint(self, path):
        self.start_epoch = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler, self.device
        )
        print(f"Resumed Tagger from epoch {self.start_epoch}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for batch in self.train_loader:
            # Unpack batch
            token_ids, char_ids, class_ids, mask = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()

            # Forward Pass
            logits = self.model(token_ids, char_ids, mask)

            # Reshape for Loss: Flatten Batch and Seq dimensions
            # logits: (B, S, C) -> (B*S, C)
            # targets: (B, S) -> (B*S)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = class_ids.view(-1)

            loss = self.criterion(logits_flat, targets_flat)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # Calculate Accuracy (ignoring padding)
            acc = calculate_accuracy(logits_flat, targets_flat, ignore_index=-100)
            total_acc += acc
            num_batches += 1

        return total_loss / num_batches, total_acc / num_batches

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                token_ids, char_ids, class_ids, mask = [
                    x.to(self.device) for x in batch
                ]

                logits = self.model(token_ids, char_ids, mask)

                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = class_ids.view(-1)

                loss = self.criterion(logits_flat, targets_flat)
                total_loss += loss.item()

                acc = calculate_accuracy(logits_flat, targets_flat, ignore_index=-100)
                total_acc += acc
                num_batches += 1

        return total_loss / num_batches, total_acc / num_batches

    def fit(self, epochs=Config.EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE):
        print(f"Starting Tagger training for {epochs} epochs...")
        patience_counter = 0

        for epoch in range(self.start_epoch, epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()

            self.scheduler.step(val_acc)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
                f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
            )

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    Config.TAGGER_MODEL_PATH,
                )
                print(f"New best model saved with Val Acc: {val_acc:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    def predict(self, token_ids, char_ids, mask):
        """Returns predicted class indices for a batch."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(token_ids, char_ids, mask)
            preds = torch.argmax(logits, dim=-1)
        return preds


class Seq2SeqEngine:
    """
    Engine for training and evaluating the Transformer Seq2Seq Fallback Model.
    """

    def __init__(self, model, device, train_loader, val_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Ignore padding (0) in loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1, verbose=True
        )

        self.best_loss = float("inf")
        self.start_epoch = 0

    def load_checkpoint(self, path):
        self.start_epoch = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler, self.device
        )
        print(f"Resumed Seq2Seq from epoch {self.start_epoch}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in self.train_loader:
            # Unpack: src, tgt, class_ids, src_mask, tgt_mask
            src, tgt, class_ids, src_mask, tgt_mask = [x.to(self.device) for x in batch]

            # Teacher Forcing Setup
            # Input to Decoder: Tgt sequence excluding the last token (EOS)
            dec_input = tgt[:, :-1]
            # Target for Loss: Tgt sequence excluding the first token (SOS)
            targets = tgt[:, 1:]

            # Adjust mask for decoder input (remove last column)
            dec_mask = tgt_mask[:, :-1]

            self.optimizer.zero_grad()

            # Forward Pass
            logits = self.model(
                src=src,
                tgt=dec_input,
                class_ids=class_ids,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=dec_mask,
            )

            # Reshape
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)

            loss = self.criterion(logits_flat, targets_flat)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src, tgt, class_ids, src_mask, tgt_mask = [
                    x.to(self.device) for x in batch
                ]

                dec_input = tgt[:, :-1]
                targets = tgt[:, 1:]
                dec_mask = tgt_mask[:, :-1]

                logits = self.model(
                    src=src,
                    tgt=dec_input,
                    class_ids=class_ids,
                    src_key_padding_mask=src_mask,
                    tgt_key_padding_mask=dec_mask,
                )

                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = targets.reshape(-1)

                loss = self.criterion(logits_flat, targets_flat)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(self, epochs=Config.EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE):
        print(f"Starting Seq2Seq training for {epochs} epochs...")
        patience_counter = 0

        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    Config.SEQ2SEQ_MODEL_PATH,
                )
                print(f"New best model saved with Val Loss: {val_loss:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    def generate_batch(self, src, class_ids, vocab, max_len=Config.MAX_CHAR_LEN):
        """
        Greedy decoding for a batch of source sequences.
        """
        self.model.eval()
        batch_size = src.size(0)
        device = src.device

        # Start symbol
        sos_id = vocab.char2id[Config.SOS_TOKEN]
        eos_id = vocab.char2id[Config.EOS_TOKEN]

        # Initial decoder input: (B, 1) filled with SOS
        tgt = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)

        # Finished flags
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        with torch.no_grad():
            for _ in range(max_len):
                # Forward pass
                # Note: src_key_padding_mask should be handled if src has padding.
                # Assuming src is (B, S) and 0 is pad.
                src_mask = src == 0

                logits = self.model(
                    src=src,
                    tgt=tgt,
                    class_ids=class_ids,
                    src_key_padding_mask=src_mask,
                    tgt_key_padding_mask=None,  # No mask needed for generated part so far
                )

                # Get last token predictions: (B, Vocab)
                next_token_logits = logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)  # (B,)

                # Append to tgt
                tgt = torch.cat([tgt, next_tokens.unsqueeze(1)], dim=1)

                # Update finished status
                finished |= next_tokens == eos_id

                if finished.all():
                    break

        # Decode to strings
        results = []
        tgt_cpu = tgt.cpu().numpy()
        for i in range(batch_size):
            # Skip SOS (index 0)
            indices = tgt_cpu[i][1:]
            chars = []
            for idx in indices:
                if idx == eos_id:
                    break
                # Also skip PAD or UNK if they appear? Usually just EOS stops it.
                if idx in vocab.id2char:
                    chars.append(vocab.id2char[idx])
            results.append("".join(chars))

        return results


def generate_submission(tagger_engine, seq2seq_engine, test_loader, kb, vocab):
    """
    Generates the submission file by combining Tagger, Knowledge Base, and Seq2Seq Fallback.
    """
    print("Generating submission...")
    tagger_engine.model.eval()
    seq2seq_engine.model.eval()

    results = []

    # We iterate the test loader
    # batch: padded_tokens, padded_chars, mask, raw_tokens_list, row_ids_list
    for batch_idx, batch in enumerate(test_loader):
        token_ids, padded_chars, mask, raw_tokens_list, row_ids_list = batch
        token_ids = token_ids.to(tagger_engine.device)
        padded_chars = padded_chars.to(tagger_engine.device)
        mask = mask.to(tagger_engine.device)

        # 1. Predict Classes
        pred_class_indices = tagger_engine.predict(token_ids, padded_chars, mask)
        pred_class_indices = pred_class_indices.cpu().numpy()

        # Collect fallback candidates for batch processing
        fallback_indices = []  # List of (batch_idx_in_batch, seq_idx)
        fallback_src = []
        fallback_class_ids = []

        batch_results = []  # Placeholder for this batch

        # 2. Iterate tokens to check KB
        batch_size = token_ids.size(0)
        for i in range(batch_size):
            sent_len = len(raw_tokens_list[i])
            sent_preds = []

            for j in range(sent_len):
                raw_token = raw_tokens_list[i][j]
                row_id = row_ids_list[i][j]
                class_idx = pred_class_indices[i, j]
                class_name = vocab.id2class.get(class_idx, "PLAIN")

                # Strategy:
                # If PLAIN/PUNCT -> Copy raw (mostly)
                # Else -> Lookup KB
                # If KB miss -> Seq2Seq

                normalized_text = None

                # Check KB
                normalized_text = kb.get(raw_token, class_name)

                if normalized_text is None:
                    if class_name == "PLAIN" or class_name == "PUNCT":
                        normalized_text = raw_token
                    else:
                        # Mark for fallback
                        fallback_indices.append((i, j))
                        # Prepare Source for Seq2Seq: (W,) tensor
                        # padded_chars[i, j] is the char sequence for this token
                        fallback_src.append(padded_chars[i, j])
                        fallback_class_ids.append(class_idx)
                        normalized_text = "<PENDING>"  # Placeholder

                sent_preds.append({"id": row_id, "after": normalized_text})

            batch_results.append(sent_preds)

        # 3. Run Seq2Seq for Fallbacks
        if fallback_src:
            # Stack sources: (N_fallback, W)
            src_tensor = torch.stack(fallback_src).to(seq2seq_engine.device)
            class_tensor = torch.tensor(fallback_class_ids, dtype=torch.long).to(
                seq2seq_engine.device
            )

            generated_texts = seq2seq_engine.generate_batch(
                src_tensor, class_tensor, vocab
            )

            # Fill back placeholders
            for idx, (b_i, s_j) in enumerate(fallback_indices):
                batch_results[b_i][s_j]["after"] = generated_texts[idx]

        # 4. Flatten and Store
        for sent_preds in batch_results:
            results.extend(sent_preds)

    # Write to CSV
    df_sub = pd.DataFrame(results)
    # Ensure columns order
    df_sub = df_sub[["id", "after"]]

    # Post-processing: Handle quotes if necessary (standard CSV handles this)
    # The requirement says: 0_0,"the"
    # Pandas to_csv handles quoting.

    df_sub.to_csv(
        Config.SUBMISSION_PATH, index=False, quoting=1
    )  # quote all non-numeric? or minimal?
    # Default quoting is usually fine, but let's ensure text is quoted if it contains special chars.
    # quoting=1 is QUOTE_ALL? No, csv.QUOTE_ALL is 1.
    # Let's stick to default pandas behavior which matches the sample usually.

    print(f"Submission saved to {Config.SUBMISSION_PATH} with {len(df_sub)} rows.")
