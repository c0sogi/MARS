import torch
import torch.nn as nn
import torchvision.models as models
import math
from library.config import Config


class ResNetEncoder(nn.Module):
    """
    ResNet-18 based visual encoder.
    Removes pooling and FC layers to preserve spatial features.
    Projects feature maps to d_model.
    """

    def __init__(self, d_model):
        super().__init__()
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=Config.ENCODER_PRETRAINED)

        # Adapt first conv layer for 1-channel input (grayscale)
        # We average the weights of the original 3 channels to initialize the 1 channel kernel
        original_weights = resnet.conv1.weight.data.clone()
        new_weights = original_weights.mean(dim=1, keepdim=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data = new_weights

        # Remove avgpool and fc
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # ResNet18 layer4 output channels = 512
        self.projection = nn.Conv2d(512, d_model, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Images (B, 1, H, W)
        Returns:
            torch.Tensor: Encoded features (S, B, E) where S = H'*W'
        """
        x = self.backbone(x)  # (B, 512, H/32, W/32)
        x = self.projection(x)  # (B, d_model, H/32, W/32)

        # Flatten spatial dimensions
        x = x.flatten(2)  # (B, d_model, S)

        # Permute to (S, B, d_model) for Transformer
        x = x.permute(2, 0, 1)
        return x


class PositionalEncoding(nn.Module):
    """
    Learnable 1D positional encoding.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
        nn.init.normal_(self.pe, mean=0, std=0.02)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (S, B, E)
        Returns:
            torch.Tensor: Positional encoded tensor (S, B, E)
        """
        # Add positional encoding up to the current sequence length
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class CNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer architecture for Image-to-Text tasks.
    """

    def __init__(self):
        super().__init__()

        # 1. Visual Encoder
        self.encoder = ResNetEncoder(Config.D_MODEL)

        # 2. Positional Encodings
        # Visual sequence length for 256x256 image through ResNet18 is 8x8=64.
        # We allocate enough buffer.
        self.vis_pos_enc = PositionalEncoding(Config.D_MODEL, max_len=256)
        self.txt_pos_enc = PositionalEncoding(
            Config.D_MODEL, max_len=Config.MAX_LEN + 50
        )

        # 3. Embeddings
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.D_MODEL)

        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=Config.NUM_DECODER_LAYERS
        )

        # 5. Prediction Head
        self.fc_out = nn.Linear(Config.D_MODEL, Config.VOCAB_SIZE)

        self.dropout = nn.Dropout(Config.DROPOUT)

    def generate_square_subsequent_mask(self, sz):
        """
        Generates a causal mask for the decoder.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, images, target_seq=None):
        """
        Forward pass for training.

        Args:
            images (torch.Tensor): (B, 1, H, W)
            target_seq (torch.Tensor, optional): (B, L) Ground truth sequences with SOS/EOS/PAD.

        Returns:
            torch.Tensor: Logits (B, L-1, VocabSize)
        """
        # --- Encode ---
        memory = self.encoder(images)  # (S_mem, B, E)
        memory = self.vis_pos_enc(memory)

        if target_seq is not None:
            # --- Decode (Training with Teacher Forcing) ---

            # Input to decoder: <SOS> ... <last_char> (exclude last token of target which is usually padding or EOS)
            # We want to predict the next token.
            # If target is [SOS, A, B, EOS, PAD], input is [SOS, A, B, EOS], output trained against [A, B, EOS, PAD]
            # Actually, usually input is [SOS, A, B] and target is [A, B, EOS].
            # Let's assume standard autoregressive setup:
            # Input: target_seq[:, :-1]
            # Target for loss: target_seq[:, 1:]

            tgt_inp = target_seq[:, :-1]  # (B, L-1)
            tgt_inp = tgt_inp.permute(1, 0)  # (L-1, B)

            # Embedding + Positional Encoding
            tgt_emb = self.embedding(tgt_inp) * math.sqrt(Config.D_MODEL)
            tgt_emb = self.txt_pos_enc(tgt_emb)
            tgt_emb = self.dropout(tgt_emb)

            # Masks
            device = images.device
            seq_len = tgt_inp.size(0)
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)

            # Padding mask (B, L-1)
            # True where padding exists
            tgt_key_padding_mask = tgt_inp.transpose(0, 1) == Config.PAD_IDX

            # Transformer Decoder
            output = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )  # (L-1, B, E)

            logits = self.fc_out(output)  # (L-1, B, V)
            return logits.transpose(0, 1)  # (B, L-1, V)

        return None

    def predict(self, images, max_len=Config.MAX_LEN, device="cuda"):
        """
        Inference using greedy decoding.

        Args:
            images (torch.Tensor): (B, 1, H, W)
            max_len (int): Maximum generation length.
            device (str): Device string.

        Returns:
            torch.Tensor: Predicted indices (B, L)
        """
        self.eval()
        B = images.size(0)

        with torch.no_grad():
            # Encode
            memory = self.encoder(images)
            memory = self.vis_pos_enc(memory)

            # Initialize input with SOS
            tgt_inp = torch.full(
                (1, B), Config.SOS_IDX, dtype=torch.long, device=device
            )

            # Keep track of finished sequences
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            predictions = []

            for _ in range(max_len):
                # Embed current sequence
                tgt_emb = self.embedding(tgt_inp) * math.sqrt(Config.D_MODEL)
                tgt_emb = self.txt_pos_enc(tgt_emb)

                # Causal mask
                tgt_mask = self.generate_square_subsequent_mask(tgt_inp.size(0)).to(
                    device
                )

                # Decode
                output = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)

                # Get logits for the last token
                last_output = output[-1, :, :]  # (B, E)
                logits = self.fc_out(last_output)  # (B, V)

                # Greedy choice
                next_token = torch.argmax(logits, dim=-1)  # (B,)

                # Store prediction
                predictions.append(next_token)

                # Update input for next step
                tgt_inp = torch.cat([tgt_inp, next_token.unsqueeze(0)], dim=0)

                # Check EOS
                is_eos = next_token == Config.EOS_IDX
                finished = finished | is_eos

                if finished.all():
                    break

            # Stack predictions (S, B) -> (B, S)
            if predictions:
                predictions = torch.stack(predictions, dim=1)
            else:
                # Edge case: max_len=0
                predictions = torch.zeros((B, 0), dtype=torch.long, device=device)

            return predictions
