from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ResidualBlock1D(nn.Module):
    """
    Residual block for 1D CNN with skip connections.
    """

    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int = 3, dropout: float = 0.2):
        """
        Initialize residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            dropout: Dropout probability
        """
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.

        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)

        Returns:
            Output tensor with residual connection
        """
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out)

        return out


class MultiScaleConv1D(nn.Module):
    """
    Multi-scale 1D convolution with different kernel sizes.
    """

    def __init__(self, in_channels: int, out_channels: int,
                kernel_sizes: List[int] = [3, 5, 7],
                dropout: float = 0.2):
        """
        Initialize multi-scale convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per kernel
            kernel_sizes: List of kernel sizes for multi-scale extraction
            dropout: Dropout probability
        """
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList()

        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.convs.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale convolutions.

        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)

        Returns:
            Concatenated multi-scale features
        """
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))

        # Concatenate along channel dimension
        return torch.cat(outputs, dim=1)


class CNN1DFeatureExtractor(nn.Module):
    """
    1D CNN for extracting short-term patterns from financial time series.

    Architecture:
    - Multi-scale initial convolution
    - Stack of residual blocks with increasing filters
    - Global pooling for fixed-size output
    - Fully connected layers for feature transformation
    """

    def __init__(self, input_dim: int, sequence_length: int,
                filters: List[int] = [64, 128, 256],
                kernel_sizes: List[int] = [3, 5, 7],
                dropout: float = 0.3,
                output_dim: int = 256):
        """
        Initialize CNN feature extractor.

        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequences
            filters: List of filter counts for each layer
            kernel_sizes: Kernel sizes for multi-scale convolution
            dropout: Dropout probability
            output_dim: Dimension of output features
        """
        super().__init__()

        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.filters = filters
        self.kernel_sizes = kernel_sizes

        # Input projection (feature dimension -> channels)
        self.input_projection = nn.Conv1d(input_dim, filters[0], 1)

        # Multi-scale initial convolution
        self.multi_scale = MultiScaleConv1D(
            filters[0], filters[0] // len(kernel_sizes),
            kernel_sizes, dropout
        )

        # Calculate total channels after multi-scale
        multi_scale_channels = filters[0] // len(kernel_sizes) * len(kernel_sizes)

        # Residual blocks
        self.residual_blocks = nn.ModuleList()

        in_channels = multi_scale_channels
        for out_channels in filters:
            self.residual_blocks.append(
                ResidualBlock1D(in_channels, out_channels, 3, dropout)
            )
            in_channels = out_channels

        # Global pooling layers
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Final projection
        # *2 for concatenated max and avg pooling
        final_channels = filters[-1] * 2
        self.fc_layers = nn.Sequential(
            nn.Linear(final_channels, output_dim),
            nn.LayerNorm(output_dim),  # Changed from BatchNorm1d to LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        # Attention weights for temporal importance
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(filters[-1], 1, 1),
            nn.Softmax(dim=-1)
        )

        logger.info(f"Initialized CNN1D with {len(filters)} layers, output dim {output_dim}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CNN feature extractor.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)

        Returns:
            Tuple of (features, attention_weights)
            - features: Extracted features (batch_size, output_dim)
            - attention_weights: Temporal attention weights (batch_size, sequence_length)
        """
        # Reshape input: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_projection(x)

        # Multi-scale convolution
        x = self.multi_scale(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Compute temporal attention
        attention_weights = self.temporal_attention(x)

        # Apply attention
        x_attended = x * attention_weights

        # Global pooling
        max_pooled = self.global_max_pool(x_attended).squeeze(-1)
        avg_pooled = self.global_avg_pool(x_attended).squeeze(-1)

        # Concatenate pooled features
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)

        # Final transformation
        features = self.fc_layers(pooled)

        # Return features and attention weights for visualization
        return features, attention_weights.squeeze(1)

    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract intermediate feature maps for visualization.

        Args:
            x: Input tensor

        Returns:
            List of feature maps from each layer
        """
        feature_maps = []

        # Reshape input
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_projection(x)
        feature_maps.append(x.clone())

        # Multi-scale convolution
        x = self.multi_scale(x)
        feature_maps.append(x.clone())

        # Residual blocks
        for i, block in enumerate(self.residual_blocks):
            x = block(x)
            feature_maps.append(x.clone())

        return feature_maps


class DilatedCNN1D(nn.Module):
    """
    Dilated CNN for capturing patterns at different time scales.
    """

    def __init__(self, input_dim: int, sequence_length: int,
                filters: int = 64, num_layers: int = 4,
                dropout: float = 0.2, output_dim: int = 128):
        """
        Initialize dilated CNN.

        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequences
            filters: Number of filters per layer
            num_layers: Number of dilated convolution layers
            dropout: Dropout probability
            output_dim: Output dimension
        """
        super().__init__()

        self.input_projection = nn.Conv1d(input_dim, filters, 1)

        # Dilated convolutions with exponentially increasing dilation
        self.dilated_convs = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            padding = dilation  # Same padding for causal convolution

            conv = nn.Sequential(
                nn.Conv1d(filters, filters, 3, dilation=dilation, padding=padding),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.dilated_convs.append(conv)

        # Global pooling and output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(filters, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dilated CNN.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)

        Returns:
            Output features (batch_size, output_dim)
        """
        # Reshape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_projection(x)

        # Apply dilated convolutions with residual connections
        for conv in self.dilated_convs:
            residual = x
            x = conv(x) + residual

        # Global pooling
        x = self.global_pool(x).squeeze(-1)

        # Output transformation
        return self.output_layer(x)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence modeling.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                kernel_size: int = 3, dropout: float = 0.2):
        """
        Initialize TCN.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden dimensions for each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        num_levels = len(hidden_dims)

        for i in range(num_levels):
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            dilation = 2 ** i

            layers.append(
                ResidualBlock1D(in_channels, out_channels, kernel_size, dropout)
            )

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)

        Returns:
            Output tensor (batch_size, sequence_length, output_dim)
        """
        # Reshape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Apply TCN
        x = self.network(x)

        # Reshape back: (batch, features, seq_len) -> (batch, seq_len, features)
        return x.transpose(1, 2)