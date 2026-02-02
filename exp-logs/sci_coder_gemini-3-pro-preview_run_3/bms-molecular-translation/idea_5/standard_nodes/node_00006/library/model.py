import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils import weight_norm
from library.config import Config


class Chomp1d(nn.Module):
    """
    Removes the last elements of a sequence to ensure causality in convolution.
    Since PyTorch Conv1d with padding adds padding to both sides, we need to
    remove the padding from the 'future' side to keep the convolution causal.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    A single block of the Temporal Convolutional Network.
    Consists of two dilated causal convolutions with ReLU, Dropout, and a residual connection.
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()

        # First dilated convolution
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated convolution
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        # Residual connection: if input and output channels differ, project input
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    The TCN decoder consisting of a stack of TemporalBlocks with exponentially increasing dilation.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Padding is calculated to maintain sequence length after chomp
            # padding = (kernel_size - 1) * dilation
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResNetEncoder(nn.Module):
    """
    ResNet-34 encoder to extract a global context vector from the image.
    Outputs a 512-dimensional vector.
    """

    def __init__(self, pretrained=True):
        super(ResNetEncoder, self).__init__()
        # Use weights parameter for modern torchvision versions
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Remove the final fully connected layer (fc)
        # Keep everything up to avgpool.
        # resnet.children() returns: conv1, bn1, relu, maxpool, layer1..4, avgpool, fc
        # We take everything except the last one (fc).
        # The output of avgpool is (B, 512, 1, 1).
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        # Flatten: (B, 512, 1, 1) -> (B, 512)
        x = x.view(x.size(0), -1)
        return x


class ResNetTCN(nn.Module):
    """
    End-to-end model combining ResNet encoder and TCN decoder.
    """

    def __init__(self, vocab_size):
        super(ResNetTCN, self).__init__()

        # Encoder
        self.encoder = ResNetEncoder(pretrained=Config.ENCODER_PRETRAINED)

        # Embedding for the text sequence
        self.embedding = nn.Embedding(vocab_size, Config.EMBEDDING_DIM)

        # TCN Decoder
        # Input dimension is Embedding Dim + Encoder Context Dim (concatenated)
        tcn_input_dim = Config.EMBEDDING_DIM + Config.ENCODER_DIM

        self.tcn = TemporalConvNet(
            num_inputs=tcn_input_dim,
            num_channels=Config.TCN_NUM_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.TCN_DROPOUT,
        )

        # Final classification layer mapping TCN output to vocabulary
        self.decoder = nn.Linear(Config.TCN_NUM_CHANNELS[-1], vocab_size)

    def forward(self, images, captions):
        """
        Forward pass for training.

        Args:
            images: (B, C, H, W)
            captions: (B, L) - Integer sequences (indices)

        Returns:
            outputs: (B, L, VocabSize) - Logits for next token prediction
        """
        # 1. Encode Image
        # (B, 512)
        features = self.encoder(images)

        # 2. Embed Captions
        # (B, L, 256)
        embeddings = self.embedding(captions)

        # 3. Condition TCN on Image
        # We repeat the image features for every time step and concatenate them with embeddings.
        # features: (B, 512) -> (B, 1, 512) -> (B, L, 512)
        features_repeated = features.unsqueeze(1).expand(-1, embeddings.size(1), -1)

        # Concatenate along channel dimension (dim 2)
        # (B, L, 256 + 512) = (B, L, 768)
        tcn_input = torch.cat((embeddings, features_repeated), dim=2)

        # TCN expects (B, Channels, Length)
        tcn_input = tcn_input.permute(0, 2, 1)

        # 4. Pass through TCN
        # (B, Hidden, L)
        y = self.tcn(tcn_input)

        # 5. Decode to Vocab
        # Permute back to (B, L, Hidden)
        y = y.permute(0, 2, 1)
        out = self.decoder(y)

        return out

    def predict(self, image, tokenizer, max_len=None, device=None):
        """
        Inference method using greedy decoding.

        Args:
            image: (C, H, W) tensor
            tokenizer: Tokenizer instance
            max_len: Maximum sequence length
            device: torch device

        Returns:
            str: Predicted InChI string
        """
        if max_len is None:
            max_len = Config.MAX_LEN
        if device is None:
            device = Config.DEVICE

        self.eval()
        with torch.no_grad():
            # Prepare image
            if image.dim() == 3:
                image = image.unsqueeze(0)  # (1, C, H, W)
            image = image.to(device)

            # Encode image once
            features = self.encoder(image)  # (1, 512)

            # Initialize sequence with SOS token
            seq = [tokenizer.SOS_IDX]

            for _ in range(max_len):
                # Prepare input sequence tensor
                input_tensor = (
                    torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
                )  # (1, L)

                # Embed
                embeddings = self.embedding(input_tensor)  # (1, L, 256)

                # Expand features to match current sequence length
                features_repeated = features.unsqueeze(1).expand(
                    -1, embeddings.size(1), -1
                )

                # Concatenate
                tcn_input = torch.cat(
                    (embeddings, features_repeated), dim=2
                )  # (1, L, 768)
                tcn_input = tcn_input.permute(0, 2, 1)  # (1, 768, L)

                # Run TCN
                # Note: We re-run the whole sequence every step. TCN is fast enough for this.
                output = self.tcn(tcn_input)  # (1, Hidden, L)

                # Get output for the last time step
                last_output = output[:, :, -1]  # (1, Hidden)
                logits = self.decoder(last_output)  # (1, Vocab)

                # Greedy selection
                probs = torch.softmax(logits, dim=1)
                next_token = torch.argmax(probs, dim=1).item()

                seq.append(next_token)

                # Stop if EOS is generated
                if next_token == tokenizer.EOS_IDX:
                    break

            return tokenizer.sequence_to_text(seq)
