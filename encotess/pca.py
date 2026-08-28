"""Fixed global PCA projection of the 1536-d latents (numpy only, no sklearn).

A single unsupervised global PCA (full-rank, 1536 components, unwhitened; fit once
over all released encodings) re-expresses every 1536-d latent in a decorrelated,
variance-ordered basis. It is a lossless orthonormal rotation of the standardized
latent (invertible back to it), and is what the package ships as the PCA-space
encodings. ``transform`` reproduces ``sklearn.decomposition.PCA.transform`` for the
fitted artifact in ``global_pca.npz``:

    X_norm = (X - X_mean) / (X_std + 1e-8)        # per-feature standardization
    Z      = (X_norm - pca_mean) @ components.T   # projection (keep dim comps)
    Z      = Z / sqrt(explained_variance)         # only if whiten (default off)
"""
from __future__ import annotations

import numpy as np


class GlobalPCA:
    """Apply a pre-fit global PCA to standardized latents."""

    def __init__(self, X_mean, X_std, pca_mean, components,
                 explained_variance=None, whiten=False):
        self.X_mean = np.asarray(X_mean, dtype=np.float64)
        self.X_std = np.asarray(X_std, dtype=np.float64)
        self.pca_mean = np.asarray(pca_mean, dtype=np.float64)
        self.components = np.asarray(components, dtype=np.float64)  # (n_comp, n_feat)
        self.explained_variance = (
            None if explained_variance is None
            else np.asarray(explained_variance, dtype=np.float64)
        )
        self.whiten = bool(whiten)

    @classmethod
    def from_npz(cls, path):
        """Load from a ``global_pca_d*.npz`` artifact."""
        z = np.load(path, allow_pickle=False)
        return cls(
            X_mean=z['X_mean'],
            X_std=z['X_std'],
            pca_mean=z['pca_mean'],
            components=z['pca_components'],
            explained_variance=z['pca_explained_variance']
            if 'pca_explained_variance' in z.files else None,
            whiten=bool(z['pca_whiten']) if 'pca_whiten' in z.files else False,
        )

    @property
    def n_components(self) -> int:
        return self.components.shape[0]

    def transform(self, X, dim: int = None) -> np.ndarray:
        """Project (N, 1536) latents to (N, dim) PCA features.

        Args:
            X: (N, n_features) raw latents.
            dim: keep only the first ``dim`` components (default: all available).
        """
        X = np.asarray(X, dtype=np.float64)
        X_norm = (X - self.X_mean) / (self.X_std + 1e-8)
        comp = self.components if dim is None else self.components[:dim]
        Z = (X_norm - self.pca_mean) @ comp.T
        if self.whiten:
            ev = self.explained_variance
            if ev is None:
                raise ValueError("whiten=True requires explained_variance")
            Z = Z / np.sqrt(ev[:comp.shape[0]])
        return Z.astype(np.float32)
