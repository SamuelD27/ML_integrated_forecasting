from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import logging
import math

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """
    Attention mechanism for sequence models.
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        """
        Initialize attention layer.

        Args:
            hidden_dim: Dimension of hidden states
            attention_dim: Dimension of attention projection
        """
        super().__init__()

        self.attention_dim = attention_dim

        # Attention parameters
        self.W_query = nn.Linear(hidden_dim, attention_dim)
        self.W_key = nn.Linear(hidden_dim, attention_dim)
        self.W_value = nn.Linear(hidden_dim, attention_dim)

        # Output projection
        self.W_out = nn.Linear(attention_dim, hidden_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.scale = math.sqrt(attention_dim)

    def forward(self, hidden_states: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism.

        Args:
            hidden_states: Sequence of hidden states (batch, seq_len, hidden_dim)
            mask: Optional mask for padding (batch, seq_len)

        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Compute query, key, value
        Q = self.W_query(hidden_states)  # (batch, seq_len, attention_dim)
        K = self.W_key(hidden_states)
        V = self.W_value(hidden_states)

        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attended = torch.bmm(attention_weights, V)  # (batch, seq_len, attention_dim)

        # Output projection
        output = self.W_out(attended)

        # Residual connection and layer norm
        output = self.layer_norm(output + hidden_states)

        return output, attention_weights


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder with attention for sequence modeling.

    Architecture:
    - Multi-layer bidirectional LSTM
    - Self-attention mechanism
    - Layer normalization
    - Configurable hidden dimensions
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                num_layers: int = 2, bidirectional: bool = True,
                dropout: float = 0.2, use_attention: bool = True,
                cell_type: str = 'lstm', output_dim: Optional[int] = None):
        """
        Initialize LSTM encoder.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
            cell_type: Type of RNN cell ('lstm' or 'gru')
            output_dim: Output dimension (default: hidden_dim * 2 if bidirectional)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.cell_type = cell_type.lower()

        # Calculate actual hidden dimension
        self.num_directions = 2 if bidirectional else 1
        self.actual_hidden_dim = hidden_dim * self.num_directions

        # Choose RNN cell type
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                input_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                input_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        # Layer normalization for each direction
        self.layer_norm = nn.LayerNorm(self.actual_hidden_dim)

        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(self.actual_hidden_dim)

        # Output projection
        self.output_dim = output_dim or self.actual_hidden_dim

        # Combine projection for the 3 strategies (mean, max, last)
        self.combine_projection = nn.Linear(
            self.actual_hidden_dim * 3, self.actual_hidden_dim
        )

        if self.output_dim != self.actual_hidden_dim:
            self.output_projection = nn.Linear(self.actual_hidden_dim, self.output_dim)
        else:
            self.output_projection = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"Initialized {cell_type.upper()} encoder: "
            f"layers={num_layers}, hidden={hidden_dim}, "
            f"bidirectional={bidirectional}, attention={use_attention}"
        )

    def forward(self, x: torch.Tensor,
               lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM encoder.

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            lengths: Actual lengths of sequences for packing (optional)

        Returns:
            Tuple of (output, hidden_states)
            - output: Encoded sequence (batch_size, output_dim)
            - hidden_states: All hidden states (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Pack sequences if lengths provided (for efficiency)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # Forward through RNN
        if self.cell_type == 'lstm':
            rnn_output, (hidden, cell) = self.rnn(x)
        else:  # GRU
            rnn_output, hidden = self.rnn(x)

        # Unpack if packed
        if lengths is not None:
            rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )

        # Apply layer normalization
        rnn_output = self.layer_norm(rnn_output)

        # Apply attention if enabled
        if self.use_attention:
            # Create mask based on lengths if provided
            mask = None
            if lengths is not None:
                mask = torch.zeros(batch_size, seq_len, device=x.device)
                for i, length in enumerate(lengths):
                    mask[i, :length] = 1

            attended_output, attention_weights = self.attention(rnn_output, mask)

            # Use attended output for final representation
            sequence_output = attended_output
        else:
            sequence_output = rnn_output
            attention_weights = None

        # Get final output (several strategies)
        # Strategy 1: Mean pooling over sequence
        output_mean = sequence_output.mean(dim=1)

        # Strategy 2: Max pooling over sequence
        output_max, _ = sequence_output.max(dim=1)

        # Strategy 3: Last hidden state (considering bidirectionality)
        if self.bidirectional:
            # Concatenate forward and backward last states
            forward_last = sequence_output[:, -1, :self.hidden_dim]
            backward_last = sequence_output[:, 0, self.hidden_dim:]
            output_last = torch.cat([forward_last, backward_last], dim=1)
        else:
            output_last = sequence_output[:, -1, :]

        # Combine strategies (concatenate and project)
        combined = torch.cat([output_mean, output_max, output_last], dim=1)

        # Project to output dimension
        output = self.combine_projection(combined)

        # Apply dropout
        output = self.dropout(output)

        # Apply output projection
        output = self.output_projection(output)

        return output, sequence_output


class StackedLSTM(nn.Module):
    """
    Stacked LSTM with skip connections between layers.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                dropout: float = 0.2, use_residual: bool = True):
        """
        Initialize stacked LSTM.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions for each layer
            dropout: Dropout probability
            use_residual: Whether to use residual connections
        """
        super().__init__()

        self.use_residual = use_residual
        self.layers = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(
                nn.LSTM(prev_dim, hidden_dim, batch_first=True)
            )
            prev_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dims[-1]

        # Residual projections if dimensions don't match
        if use_residual:
            self.residual_projections = nn.ModuleList()
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                if prev_dim != hidden_dim:
                    self.residual_projections.append(
                        nn.Linear(prev_dim, hidden_dim)
                    )
                else:
                    self.residual_projections.append(nn.Identity())
                prev_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stacked LSTM.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Output tensor (batch, seq_len, output_dim)
        """
        for i, lstm in enumerate(self.layers):
            residual = x

            # Forward through LSTM
            x, _ = lstm(x)
            x = self.dropout(x)

            # Apply residual connection if enabled
            if self.use_residual:
                residual = self.residual_projections[i](residual)
                x = x + residual

        return x


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM for spatiotemporal pattern extraction.
    Useful when treating multiple features as spatial dimensions.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                kernel_size: int = 3, num_layers: int = 1):
        """
        Initialize ConvLSTM.

        Args:
            input_dim: Number of input channels
            hidden_dim: Number of hidden channels
            kernel_size: Size of convolutional kernel
            num_layers: Number of ConvLSTM layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.padding = kernel_size // 2

        # Create ConvLSTM cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(
                self._make_convlstm_cell(in_dim, hidden_dim)
            )

    def _make_convlstm_cell(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Create a single ConvLSTM cell."""
        return nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ConvLSTM.

        Args:
            x: Input tensor (batch, seq_len, features)

        Returns:
            Output tensor (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden states
        h = [torch.zeros(batch_size, self.hidden_dim, device=x.device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim, device=x.device)
             for _ in range(self.num_layers)]

        outputs = []

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]

            for layer_idx, cell in enumerate(self.cells):
                if layer_idx == 0:
                    h[layer_idx], c[layer_idx] = cell(x_t, (h[layer_idx], c[layer_idx]))
                else:
                    h[layer_idx], c[layer_idx] = cell(
                        h[layer_idx-1], (h[layer_idx], c[layer_idx])
                    )

            outputs.append(h[-1])

        return torch.stack(outputs, dim=1)