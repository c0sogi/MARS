import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
from library.config import Config


class Encoder(nn.Module):
    """
    Encoder module using GRU.
    Encodes the input sequence into a context vector.
    """

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=Config.PAD_IDX)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_len, emb_dim]

        outputs, hidden = self.rnn(embedded)
        # outputs: [batch_size, src_len, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]

        return hidden


class Decoder(nn.Module):
    """
    Decoder module using GRU.
    Decodes the context vector into the target sequence.
    """

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=Config.PAD_IDX)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input: [batch_size] (single token indices)
        # hidden: [n_layers, batch_size, hid_dim]

        input = input.unsqueeze(1)
        # input: [batch_size, 1]

        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, emb_dim]

        output, hidden = self.rnn(embedded, hidden)
        # output: [batch_size, 1, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]

        prediction = self.fc_out(output.squeeze(1))
        # prediction: [batch_size, output_dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model wrapping Encoder and Decoder.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode
        hidden = self.encoder(src)

        # First input to the decoder is the <SOS> token
        input = trg[:, 0]

        for t in range(1, trg_len):
            # Decode
            output, hidden = self.decoder(input, hidden)

            # Store prediction
            outputs[:, t, :] = output

            # Get the highest predicted token
            top1 = output.argmax(1)

            # Teacher forcing: use actual next token or predicted token
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1

        return outputs


class NeuralSolver:
    """
    High-level wrapper for training and inference of the Seq2Seq model.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.device = torch.device(Config.DEVICE)

        # Initialize Model Components
        input_dim = len(tokenizer)
        output_dim = len(tokenizer)

        encoder = Encoder(
            input_dim,
            Config.EMBED_DIM,
            Config.HIDDEN_DIM,
            Config.NUM_LAYERS,
            Config.DROPOUT,
        )

        decoder = Decoder(
            output_dim,
            Config.EMBED_DIM,
            Config.HIDDEN_DIM,
            Config.NUM_LAYERS,
            Config.DROPOUT,
        )

        self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)

    def train(self, train_loader, val_loader):
        """
        Trains the model with Early Stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")
        print(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()

            # Training Loop
            self.model.train()
            epoch_loss = 0

            for batch in train_loader:
                src = batch["input_ids"].to(self.device)
                trg = batch["target_ids"].to(self.device)

                self.optimizer.zero_grad()

                output = self.model(src, trg, Config.TEACHER_FORCING_RATIO)

                # Reshape for loss calculation
                # output: [batch_size, trg_len, output_dim] -> [(batch_size * trg_len) - 1, output_dim]
                # trg: [batch_size, trg_len] -> [(batch_size * trg_len) - 1]
                # We ignore the 0th index (SOS) for loss calculation usually,
                # or the model output at t corresponds to prediction for t.
                # In the loop above, outputs[:, t] is prediction for trg[:, t].
                # So we slice from 1 to end.

                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = self.criterion(output, trg)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.CLIP_GRAD
                )
                self.optimizer.step()

                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_loader)

            # Validation Loop
            val_loss = self.evaluate(val_loader)

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
            print(f"\tTrain Loss: {train_loss}")
            print(f"\t Val. Loss: {val_loss}")

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"\tSaved best model to {Config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                print(
                    f"\tNo improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def evaluate(self, loader):
        """
        Evaluates the model on a dataset.
        """
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for batch in loader:
                src = batch["input_ids"].to(self.device)
                trg = batch["target_ids"].to(self.device)

                # Turn off teacher forcing for validation
                output = self.model(src, trg, 0)

                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = self.criterion(output, trg)
                epoch_loss += loss.item()

        return epoch_loss / len(loader)

    def predict(self, loader):
        """
        Generates predictions for the given loader using greedy decoding.

        Returns:
            list: List of predicted strings.
        """
        # Load best model if available
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
            self.model.eval()
        else:
            print("Warning: No checkpoint found. Predicting with current weights.")

        predictions = []

        with torch.no_grad():
            for batch in loader:
                src = batch["input_ids"].to(self.device)
                batch_size = src.shape[0]

                # Encode
                hidden = self.model.encoder(src)

                # Initialize decoder input with SOS
                input_token = torch.tensor([Config.SOS_IDX] * batch_size).to(
                    self.device
                )

                # Store decoded indices
                decoded_indices = torch.zeros(
                    batch_size, Config.MAX_SEQ_LEN, dtype=torch.long
                ).to(self.device)

                for t in range(Config.MAX_SEQ_LEN):
                    output, hidden = self.model.decoder(input_token, hidden)
                    pred_token = output.argmax(1)

                    decoded_indices[:, t] = pred_token
                    input_token = pred_token

                # Convert indices to strings
                # We do this batch-wise on CPU
                decoded_indices = decoded_indices.cpu().tolist()
                for seq in decoded_indices:
                    pred_str = self.tokenizer.decode(seq, remove_special_tokens=True)
                    predictions.append(pred_str)

        return predictions
