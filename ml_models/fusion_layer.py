"""
Attention-based Fusion Layer
=============================

This module implements the fusion mechanism to combine outputs from
CNN, LSTM, and Transformer modules using attention-based weighting.

Features:
- Multi-modal attention fusion
- Feature importance weighting
- Gating mechanisms
- Unified representation output
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion layer for combining multiple model outputs.

    This layer takes outputs from different encoders (CNN, LSTM, Transformer)
    and combines them using learned attention weights to produce a unified
    representation.
    """

    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 256,
                output_dim: int = 256, dropout: float = 0.1,
                use_gating: bool = True):
        """
        Initialize attention fusion layer.

        Args:
            input_dims: Dictionary mapping encoder names to their output dimensions
            hidden_dim: Hidden dimension for attention computation
            output_dim: Final output dimension
            dropout: Dropout probability
            use_gating: Whether to use gating mechanism
        """
        super().__init__()

        self.encoder_names = list(input_dims.keys())
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gating = use_gating

        # Input projections for each encoder
        self.input_projections = nn.ModuleDict()
        for name, dim in input_dims.items():
            self.input_projections[name] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # Attention mechanism
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(input_dims)),
            nn.Softmax(dim=-1)
        )

        # Gating mechanism (optional)
        if use_gating:
            self.gates = nn.ModuleDict()
            for name in input_dims.keys():
                self.gates[name] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid()
                )

        # Feature importance estimation
        self.feature_importance = nn.ModuleDict()
        for name in input_dims.keys():
            self.feature_importance[name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        # Final fusion layers
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )

        # Residual connection if dims match
        self.residual_projection = None
        if all(dim == output_dim for dim in input_dims.values()):
            self.residual_projection = nn.Linear(len(input_dims) * output_dim, output_dim)

        logger.info(
            f"Initialized AttentionFusion: "
            f"inputs={list(input_dims.keys())}, "
            f"hidden={hidden_dim}, output={output_dim}"
        )

    def forward(self, encoder_outputs: Dict[str, torch.Tensor],
               return_weights: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Forward pass through fusion layer.

        Args:
            encoder_outputs: Dictionary mapping encoder names to their outputs
            return_weights: Whether to return attention weights

        Returns:
            Tuple of (fused_output, attention_weights)
            - fused_output: Unified representation (batch_size, output_dim)
            - attention_weights: Dictionary of attention weights per encoder
        """
        batch_size = next(iter(encoder_outputs.values())).size(0)

        # Project encoder outputs
        projected = {}
        for name, output in encoder_outputs.items():
            projected[name] = self.input_projections[name](output)

        # Apply gating if enabled
        if self.use_gating:
            gated = {}
            for name, proj in projected.items():
                gate = self.gates[name](proj)
                gated[name] = gate * proj
            projected = gated

        # Compute feature importance
        importance = {}
        for name, proj in projected.items():
            importance[name] = self.feature_importance[name](proj)

        # Apply feature importance weighting
        weighted = {}
        for name, proj in projected.items():
            weighted[name] = proj * importance[name]

        # Stack all encoder outputs
        stacked = torch.stack(list(weighted.values()), dim=1)  # (batch, n_encoders, hidden_dim)

        # Compute attention weights
        concatenated = torch.cat(list(weighted.values()), dim=-1)  # (batch, n_encoders * hidden_dim)
        attention_weights = self.attention_weights(concatenated)  # (batch, n_encoders)

        # Apply attention weights
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (batch, n_encoders, 1)
        attended = (stacked * attention_weights_expanded).sum(dim=1)  # (batch, hidden_dim)

        # Final fusion
        fused_output = self.fusion_network(attended)

        # Add residual connection if available
        if self.residual_projection is not None:
            residual = torch.cat(list(encoder_outputs.values()), dim=-1)
            fused_output = fused_output + self.residual_projection(residual)

        if return_weights:
            # Convert attention weights to dictionary
            weight_dict = {
                name: attention_weights[:, i].mean().item()
                for i, name in enumerate(self.encoder_names)
            }
            return fused_output, weight_dict
        else:
            return fused_output, None


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion layer that combines features at multiple levels.
    """

    def __init__(self, low_level_dim: int, mid_level_dim: int, high_level_dim: int,
                output_dim: int = 256, dropout: float = 0.1):
        """
        Initialize hierarchical fusion.

        Args:
            low_level_dim: Dimension of low-level features (CNN)
            mid_level_dim: Dimension of mid-level features (LSTM)
            high_level_dim: Dimension of high-level features (Transformer)
            output_dim: Final output dimension
            dropout: Dropout probability
        """
        super().__init__()

        # First level: Combine low and mid
        self.low_mid_fusion = nn.Sequential(
            nn.Linear(low_level_dim + mid_level_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Second level: Combine with high
        self.final_fusion = nn.Sequential(
            nn.Linear(output_dim + high_level_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention weights for each level
        self.level_attention = nn.Sequential(
            nn.Linear(output_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, low_features: torch.Tensor, mid_features: torch.Tensor,
               high_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical fusion.

        Args:
            low_features: Low-level features from CNN
            mid_features: Mid-level features from LSTM
            high_features: High-level features from Transformer

        Returns:
            Fused output tensor
        """
        # First level fusion
        low_mid_combined = torch.cat([low_features, mid_features], dim=-1)
        low_mid_fused = self.low_mid_fusion(low_mid_combined)

        # Second level fusion
        all_combined = torch.cat([low_mid_fused, high_features], dim=-1)
        final_fused = self.final_fusion(all_combined)

        # Compute level attention
        level_weights = self.level_attention(final_fused)

        # Apply weighted combination
        weighted_output = (
            level_weights[:, 0:1] * low_features +
            level_weights[:, 1:2] * mid_features +
            level_weights[:, 2:3] * high_features
        )

        return final_fused + weighted_output


class DynamicFusion(nn.Module):
    """
    Dynamic fusion layer that adapts weights based on input characteristics.
    """

    def __init__(self, input_dims: List[int], context_dim: int,
                output_dim: int = 256, dropout: float = 0.1):
        """
        Initialize dynamic fusion.

        Args:
            input_dims: List of input dimensions from different encoders
            context_dim: Dimension of context vector (e.g., market state)
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.n_inputs = len(input_dims)
        total_dim = sum(input_dims)

        # Context-aware weight generator
        self.weight_generator = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_inputs),
            nn.Softmax(dim=-1)
        )

        # Input projections
        self.projections = nn.ModuleList()
        for dim in input_dims:
            self.projections.append(
                nn.Linear(dim, output_dim)
            )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, inputs: List[torch.Tensor],
               context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dynamic fusion.

        Args:
            inputs: List of encoder outputs
            context: Context vector for weight generation

        Returns:
            Dynamically fused output
        """
        # Generate fusion weights from context
        weights = self.weight_generator(context)

        # Project inputs
        projected = []
        for i, inp in enumerate(inputs):
            projected.append(self.projections[i](inp))

        # Stack projected inputs
        stacked = torch.stack(projected, dim=1)  # (batch, n_inputs, output_dim)

        # Apply dynamic weights
        weights_expanded = weights.unsqueeze(-1)  # (batch, n_inputs, 1)
        weighted = (stacked * weights_expanded).sum(dim=1)

        # Final fusion
        return self.fusion(weighted)


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion supporting various combination strategies.
    """

    def __init__(self, input_dims: Dict[str, int], output_dim: int = 256,
                fusion_type: str = 'attention', dropout: float = 0.1):
        """
        Initialize multi-modal fusion.

        Args:
            input_dims: Dictionary of encoder names to dimensions
            output_dim: Output dimension
            fusion_type: Type of fusion ('attention', 'concatenate', 'average', 'gated')
            dropout: Dropout probability
        """
        super().__init__()

        self.fusion_type = fusion_type
        self.input_dims = input_dims
        self.output_dim = output_dim

        if fusion_type == 'attention':
            self.fusion = AttentionFusion(input_dims, output_dim, output_dim, dropout)
        elif fusion_type == 'concatenate':
            total_dim = sum(input_dims.values())
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim)
            )
        elif fusion_type == 'average':
            # Project all to same dimension then average
            self.projections = nn.ModuleDict()
            for name, dim in input_dims.items():
                self.projections[name] = nn.Linear(dim, output_dim)
        elif fusion_type == 'gated':
            self.gates = nn.ModuleDict()
            self.projections = nn.ModuleDict()
            for name, dim in input_dims.items():
                self.projections[name] = nn.Linear(dim, output_dim)
                self.gates[name] = nn.Sequential(
                    nn.Linear(dim, output_dim),
                    nn.Sigmoid()
                )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, encoder_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multi-modal fusion.

        Args:
            encoder_outputs: Dictionary of encoder outputs

        Returns:
            Fused output tensor
        """
        if self.fusion_type == 'attention':
            output, _ = self.fusion(encoder_outputs)
            return output
        elif self.fusion_type == 'concatenate':
            concatenated = torch.cat(list(encoder_outputs.values()), dim=-1)
            return self.fusion(concatenated)
        elif self.fusion_type == 'average':
            projected = []
            for name, output in encoder_outputs.items():
                projected.append(self.projections[name](output))
            return torch.stack(projected).mean(dim=0)
        elif self.fusion_type == 'gated':
            gated_outputs = []
            for name, output in encoder_outputs.items():
                gate = self.gates[name](output)
                projected = self.projections[name](output)
                gated_outputs.append(gate * projected)
            return torch.stack(gated_outputs).sum(dim=0)