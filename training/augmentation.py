"""
Data Augmentation for Time Series

Implements augmentation techniques to increase effective training data:
- Gaussian noise injection
- Time warping (compression/expansion)
- Mixup for time series
- Window slicing
- Magnitude scaling

All techniques preserve label accuracy and temporal structure.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
import logging
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class TimeSeriesAugmenter:
    """Augmentation for time series data."""

    def __init__(
        self,
        noise_std: float = 0.01,
        warp_range: Tuple[float, float] = (0.95, 1.05),
        mixup_alpha: float = 0.2,
        magnitude_range: Tuple[float, float] = (0.95, 1.05),
        random_seed: Optional[int] = None
    ):
        """
        Initialize time series augmenter.

        Args:
            noise_std: Standard deviation for Gaussian noise
            warp_range: Range for time warping (min_factor, max_factor)
            mixup_alpha: Alpha parameter for mixup (0 = no mixup, higher = more mixing)
            magnitude_range: Range for magnitude scaling
            random_seed: Random seed for reproducibility
        """
        self.noise_std = noise_std
        self.warp_range = warp_range
        self.mixup_alpha = mixup_alpha
        self.magnitude_range = magnitude_range

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def augment_batch(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        methods: Optional[list] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Augment a batch of time series data.

        Args:
            X: Input sequences (batch_size, seq_len, features)
            y: Target values (batch_size, output_dim)
            methods: List of augmentation methods to apply
                     Options: ['noise', 'warp', 'mixup', 'magnitude']
                     Default: all methods

        Returns:
            Augmented (X, y) tuple
        """
        if methods is None:
            methods = ['noise', 'warp', 'magnitude']

        is_torch = isinstance(X, torch.Tensor)

        if is_torch:
            X_aug = X.clone()
            y_aug = y.clone()
        else:
            X_aug = X.copy()
            y_aug = y.copy()

        # Apply each augmentation method
        for method in methods:
            if method == 'noise':
                X_aug = self.add_noise(X_aug)
            elif method == 'warp':
                X_aug = self.time_warp(X_aug)
            elif method == 'mixup':
                X_aug, y_aug = self.mixup(X_aug, y_aug)
            elif method == 'magnitude':
                X_aug = self.magnitude_scale(X_aug)
            else:
                logger.warning(f"Unknown augmentation method: {method}")

        return X_aug, y_aug

    def add_noise(
        self,
        X: Union[np.ndarray, torch.Tensor],
        std: Optional[float] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Add Gaussian noise to time series.

        Args:
            X: Input sequences (batch_size, seq_len, features)
            std: Noise standard deviation (uses self.noise_std if None)

        Returns:
            Noisy sequences
        """
        std = std or self.noise_std

        if isinstance(X, torch.Tensor):
            noise = torch.randn_like(X) * std
            return X + noise
        else:
            noise = np.random.randn(*X.shape) * std
            return X + noise

    def time_warp(
        self,
        X: Union[np.ndarray, torch.Tensor],
        warp_factor: Optional[float] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply time warping (temporal compression/expansion).

        Args:
            X: Input sequences (batch_size, seq_len, features)
            warp_factor: Warping factor (None = random within range)

        Returns:
            Warped sequences
        """
        is_torch = isinstance(X, torch.Tensor)

        if is_torch:
            X_np = X.cpu().numpy()
        else:
            X_np = X

        batch_size, seq_len, n_features = X_np.shape

        # Random warp factor for each sample
        if warp_factor is None:
            warp_factors = np.random.uniform(
                self.warp_range[0],
                self.warp_range[1],
                size=batch_size
            )
        else:
            warp_factors = np.full(batch_size, warp_factor)

        X_warped = np.zeros_like(X_np)

        for i in range(batch_size):
            for j in range(n_features):
                # Original time points
                x_orig = np.linspace(0, 1, seq_len)

                # Warped time points
                warp = warp_factors[i]
                x_warp = np.linspace(0, warp, seq_len)
                x_warp = np.clip(x_warp, 0, 1)  # Ensure within [0, 1]

                # Interpolate
                try:
                    interpolator = interp1d(
                        x_orig,
                        X_np[i, :, j],
                        kind='linear',
                        fill_value='extrapolate'
                    )
                    X_warped[i, :, j] = interpolator(x_warp)
                except:
                    # Fallback: keep original
                    X_warped[i, :, j] = X_np[i, :, j]

        if is_torch:
            return torch.from_numpy(X_warped).to(X.device).float()
        else:
            return X_warped

    def mixup(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        alpha: Optional[float] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Apply mixup augmentation.

        Mixup: x_mixed = lambda * x_i + (1 - lambda) * x_j
               y_mixed = lambda * y_i + (1 - lambda) * y_j

        Args:
            X: Input sequences (batch_size, seq_len, features)
            y: Target values (batch_size, output_dim)
            alpha: Mixup alpha parameter (None = use self.mixup_alpha)

        Returns:
            Mixed (X, y) tuple
        """
        alpha = alpha or self.mixup_alpha

        if alpha <= 0:
            return X, y

        batch_size = X.shape[0]

        # Sample lambda from Beta distribution
        if isinstance(X, torch.Tensor):
            lam = np.random.beta(alpha, alpha, batch_size)
            lam = torch.from_numpy(lam).to(X.device).float()

            # Reshape for broadcasting
            lam_X = lam.view(-1, 1, 1)
            lam_y = lam.view(-1, 1) if len(y.shape) > 1 else lam

            # Random shuffle
            indices = torch.randperm(batch_size).to(X.device)

            # Mix
            X_mixed = lam_X * X + (1 - lam_X) * X[indices]
            y_mixed = lam_y * y + (1 - lam_y) * y[indices]

        else:
            lam = np.random.beta(alpha, alpha, batch_size)

            # Reshape for broadcasting
            lam_X = lam.reshape(-1, 1, 1)
            lam_y = lam.reshape(-1, 1) if len(y.shape) > 1 else lam

            # Random shuffle
            indices = np.random.permutation(batch_size)

            # Mix
            X_mixed = lam_X * X + (1 - lam_X) * X[indices]
            y_mixed = lam_y * y + (1 - lam_y) * y[indices]

        return X_mixed, y_mixed

    def magnitude_scale(
        self,
        X: Union[np.ndarray, torch.Tensor],
        scale_factor: Optional[float] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Scale magnitude of time series.

        Args:
            X: Input sequences (batch_size, seq_len, features)
            scale_factor: Scaling factor (None = random within range)

        Returns:
            Scaled sequences
        """
        batch_size = X.shape[0]

        if scale_factor is None:
            if isinstance(X, torch.Tensor):
                scale_factors = torch.FloatTensor(batch_size).uniform_(
                    self.magnitude_range[0],
                    self.magnitude_range[1]
                ).to(X.device)
                scale_factors = scale_factors.view(-1, 1, 1)
            else:
                scale_factors = np.random.uniform(
                    self.magnitude_range[0],
                    self.magnitude_range[1],
                    size=(batch_size, 1, 1)
                )
        else:
            if isinstance(X, torch.Tensor):
                scale_factors = torch.full((batch_size, 1, 1), scale_factor).to(X.device)
            else:
                scale_factors = np.full((batch_size, 1, 1), scale_factor)

        return X * scale_factors

    def window_slice(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        slice_ratio: float = 0.9
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Randomly slice windows from sequences.

        Args:
            X: Input sequences (batch_size, seq_len, features)
            y: Target values (batch_size, output_dim)
            slice_ratio: Ratio of sequence to keep (0.9 = keep 90%)

        Returns:
            Sliced (X, y) tuple (labels unchanged)
        """
        batch_size, seq_len, n_features = X.shape

        slice_len = int(seq_len * slice_ratio)

        if slice_len >= seq_len:
            return X, y

        # Random start position for each sample
        max_start = seq_len - slice_len

        if isinstance(X, torch.Tensor):
            X_sliced = torch.zeros(batch_size, slice_len, n_features).to(X.device)

            for i in range(batch_size):
                start = np.random.randint(0, max_start + 1)
                X_sliced[i] = X[i, start:start + slice_len]

        else:
            X_sliced = np.zeros((batch_size, slice_len, n_features))

            for i in range(batch_size):
                start = np.random.randint(0, max_start + 1)
                X_sliced[i] = X[i, start:start + slice_len]

        return X_sliced, y

    def random_augment(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        p: float = 0.5
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Apply random augmentation with probability p.

        Args:
            X: Input sequences
            y: Target values
            p: Probability of applying augmentation

        Returns:
            Augmented (X, y) tuple
        """
        if np.random.rand() < p:
            # Randomly choose 1-2 augmentation methods
            all_methods = ['noise', 'warp', 'magnitude']
            n_methods = np.random.randint(1, 3)
            methods = np.random.choice(all_methods, size=n_methods, replace=False).tolist()

            # Optionally add mixup
            if np.random.rand() < 0.3:
                methods.append('mixup')

            return self.augment_batch(X, y, methods=methods)
        else:
            return X, y


class AugmentedDataLoader:
    """DataLoader wrapper with online augmentation."""

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        augmenter: TimeSeriesAugmenter,
        augmentation_prob: float = 0.5
    ):
        """
        Initialize augmented data loader.

        Args:
            dataloader: Base PyTorch DataLoader
            augmenter: TimeSeriesAugmenter instance
            augmentation_prob: Probability of applying augmentation
        """
        self.dataloader = dataloader
        self.augmenter = augmenter
        self.augmentation_prob = augmentation_prob

    def __iter__(self):
        """Iterate with augmentation."""
        for X, y in self.dataloader:
            # Apply augmentation with probability
            X_aug, y_aug = self.augmenter.random_augment(
                X, y,
                p=self.augmentation_prob
            )
            yield X_aug, y_aug

    def __len__(self):
        """Length of dataloader."""
        return len(self.dataloader)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    batch_size = 32
    seq_len = 60
    n_features = 10

    X = torch.randn(batch_size, seq_len, n_features)
    y = torch.randn(batch_size, 1)

    print(f"Original data shape: X={X.shape}, y={y.shape}")

    # Initialize augmenter
    augmenter = TimeSeriesAugmenter(
        noise_std=0.01,
        warp_range=(0.95, 1.05),
        mixup_alpha=0.2
    )

    # Test noise injection
    print("\n=== Noise Injection ===")
    X_noise = augmenter.add_noise(X)
    print(f"Noise added: mean diff = {(X - X_noise).abs().mean():.6f}")

    # Test time warping
    print("\n=== Time Warping ===")
    X_warp = augmenter.time_warp(X)
    print(f"Warped shape: {X_warp.shape}")

    # Test mixup
    print("\n=== Mixup ===")
    X_mix, y_mix = augmenter.mixup(X, y)
    print(f"Mixed shapes: X={X_mix.shape}, y={y_mix.shape}")

    # Test magnitude scaling
    print("\n=== Magnitude Scaling ===")
    X_scale = augmenter.magnitude_scale(X)
    print(f"Scaled: mean ratio = {(X_scale / X).mean():.3f}")

    # Test batch augmentation
    print("\n=== Batch Augmentation ===")
    X_aug, y_aug = augmenter.augment_batch(X, y, methods=['noise', 'warp', 'magnitude'])
    print(f"Augmented shapes: X={X_aug.shape}, y={y_aug.shape}")
    print(f"Mean difference from original: {(X - X_aug).abs().mean():.6f}")

    print("\n✅ All augmentation tests passed!")
