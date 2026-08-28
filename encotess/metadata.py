"""Stellar-metadata handling for the EncoTESS encoder.

Two pieces:

- ``MetadataStandardizer`` — deterministic, data-independent normalization of the
  13 raw metadata fields the encoder was trained on. Portable across datasets
  (no fitted statistics), so the same rules apply at release time.
- ``MetadataEncoder`` — the small MLP that maps the standardized metadata to the
  32-d embedding broadcast to every RNN timestep. Supports the DOROTHY-style
  binary mask channel (``use_mask=True``) the released encoder was trained with,
  which lets callers omit fields they don't have.

The field order in ``DEFAULT_METADATA_FIELDS`` is load-bearing: it is the exact
column order the trained weights expect.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# Default metadata fields, in the order the encoder was trained on.
DEFAULT_METADATA_FIELDS = [
    'cadence_s',
    'Tmag',
    'sector',
    'camera',
    'ccd',
    'parallax',
    'parallax_error',
    'G0',
    'G0_err',
    'BPRP0',
    'BPRP0_err',
    'median_flux',
    'iqr_half_flux',
]

# Astrophysical vs instrumental grouping of the fields. Not used by the released
# single-encoder path; provided for callers that want to group fields by type.
ASTRO_METADATA_FIELDS = [
    'Tmag', 'parallax', 'parallax_error', 'G0', 'G0_err',
    'BPRP0', 'BPRP0_err', 'median_flux', 'iqr_half_flux',
]
INSTRUMENTAL_METADATA_FIELDS = ['sector', 'camera', 'ccd']


class MetadataEncoder(nn.Module):
    """MLP encoder for metadata features with an optional validity-mask channel.

    When ``use_mask=True`` the encoder concatenates a binary mask to the input so
    it can distinguish "value is 0" from "value is missing". The released
    EncoTESS encoder was trained with ``use_mask=True``.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: list = None,
        dropout: float = 0.1,
        use_mask: bool = False,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.use_mask = use_mask
        self.input_dim = input_dim

        actual_input_dim = input_dim * 2 if use_mask else input_dim

        layers = []
        prev_dim = actual_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Encode metadata to the latent embedding.

        Args:
            x: (B, input_dim) standardized metadata features.
            mask: (B, input_dim) binary mask (1=valid, 0=missing). Required when
                ``use_mask=True``; if None, all values are assumed valid.
        """
        if self.use_mask:
            if mask is None:
                mask = torch.ones_like(x)
            x = torch.cat([x, mask], dim=1)
        return self.encoder(x)


class MetadataStandardizer:
    """Deterministic normalization of metadata features for encoder input.

    Uses fixed, data-independent transforms so it is portable:
        - sector: divide by 100
        - cadence_s, Tmag, parallax, parallax_error, G0, G0_err, BPRP0_err,
          median_flux, iqr_half_flux: log10 transform
        - BPRP0, camera, ccd: raw (no transformation)

    ``transform`` returns NaN-free features; when ``return_mask=True`` it also
    returns a binary validity mask (1 valid, 0 was NaN/non-positive) suitable for
    the encoder's mask channel.
    """

    LOG10_FIELDS = {
        'cadence_s',
        'Tmag',
        'parallax',
        'parallax_error',
        'G0',
        'G0_err',
        'BPRP0_err',
        'median_flux',
        'iqr_half_flux',
    }

    SCALE_FIELDS = {
        'sector': 100.0,
    }

    RAW_FIELDS = {
        'BPRP0',
        'camera',
        'ccd',
    }

    def __init__(self, fields: list = None):
        self.fields = fields if fields is not None else DEFAULT_METADATA_FIELDS

    def transform(self, data: dict, return_mask: bool = False):
        """Transform a dict of {field: array} into standardized features.

        Args:
            data: mapping field name -> array of raw values (one per light curve).
                Missing fields may be supplied as arrays of NaN; they will be
                flagged in the mask and filled with 0.
            return_mask: also return the (N, n_fields) validity mask.
        """
        n_samples = len(data[self.fields[0]])
        features = np.zeros((n_samples, len(self.fields)), dtype=np.float32)
        mask = np.ones((n_samples, len(self.fields)), dtype=np.float32)

        for i, field in enumerate(self.fields):
            values = np.array(data[field], dtype=np.float32)

            if field in self.LOG10_FIELDS:
                values = np.where(values > 0, np.log10(values), np.nan)
            elif field in self.SCALE_FIELDS:
                values = values / self.SCALE_FIELDS[field]
            # RAW_FIELDS: no transformation.

            is_missing = np.isnan(values)
            mask[:, i] = (~is_missing).astype(np.float32)
            features[:, i] = np.nan_to_num(values, nan=0.0)

        if return_mask:
            return features, mask
        return features

    def inverse_transform_field(self, field: str, values: np.ndarray) -> np.ndarray:
        """Inverse-transform a single field back to its original scale."""
        if field in self.LOG10_FIELDS:
            return 10 ** values
        elif field in self.SCALE_FIELDS:
            return values * self.SCALE_FIELDS[field]
        return values

    def get_field_info(self) -> dict:
        """Return how each field is transformed (for documentation)."""
        info = {}
        for field in self.fields:
            if field in self.LOG10_FIELDS:
                info[field] = 'log10'
            elif field in self.SCALE_FIELDS:
                info[field] = f'divide by {self.SCALE_FIELDS[field]}'
            else:
                info[field] = 'raw'
        return info
