"""Light-curve I/O helpers for EncoTESS.

Thin utilities over the HDF5 light-curve layout EncoTESS reads (datasets ``flux``,
``flux_err``, ``time``, ``length`` and a ``metadata`` group) plus in-memory array
packing. The heavy per-batch extraction loop lives in ``encode.py``; this module
only reads the lightweight identifier/metadata tables and standardizes metadata.
"""
from __future__ import annotations

import numpy as np

from encotess.metadata import DEFAULT_METADATA_FIELDS, MetadataStandardizer

# Edge samples stripped from every raw light curve before the encoder sees them
# (TESS scattered-light / thermal-settling artifacts). Must match training (10).
DEFAULT_TRIM_EDGES = 10


def read_metadata_table(h5_path, fields=None, indices=None):
    """Read the standardizer's raw metadata fields from an H5 file.

    Returns a dict {field: np.ndarray}. Missing fields are filled with zeros so
    the standardizer's mask flags them.
    """
    import h5py
    fields = fields or DEFAULT_METADATA_FIELDS
    with h5py.File(h5_path, 'r') as f:
        meta = f.get('metadata')
        n = f['flux'].shape[0] if indices is None else len(indices)
        raw = {}
        for feat in fields:
            if meta is not None and feat in meta:
                vals = meta[feat][:] if indices is None else meta[feat][indices]
                raw[feat] = vals.astype(np.float32)
            else:
                raw[feat] = np.zeros(n, dtype=np.float32)
    return raw


def read_identifiers(h5_path, indices=None):
    """Read identifiers + photometry columns that accompany the encodings.

    Returns a dict with gaia_ids (str), tic_ids, sectors, lengths, and, when
    present, bprp0/bprp0_err/mg/mg_err. Absent columns come back as NaN arrays.
    """
    import h5py
    with h5py.File(h5_path, 'r') as f:
        meta = f['metadata']
        sl = slice(None) if indices is None else indices
        n = f['flux'].shape[0] if indices is None else len(indices)

        def col(name, cast=float, default=np.nan):
            if name in meta:
                return meta[name][sl].astype(cast) if cast is not float \
                    else meta[name][sl].astype(float)
            return np.full(n, default)

        gaia_ids = np.array([g.decode('utf-8').strip() if isinstance(g, bytes) else str(g).strip()
                             for g in meta['GaiaDR3_ID'][sl]])
        out = {
            'gaia_ids': gaia_ids,
            'tic_ids': meta['tic'][sl].astype(np.int64) if 'tic' in meta
                       else np.full(n, -1, dtype=np.int64),
            'sectors': meta['sector'][sl].astype(np.int64) if 'sector' in meta
                       else np.full(n, -1, dtype=np.int64),
            'lengths': f['length'][sl].astype(np.int64),
            'bprp0': col('BPRP0'),
            'bprp0_err': col('e_BPRP0') if 'e_BPRP0' in meta else col('BPRP0_err'),
            'mg': col('MG_quick'),
            'mg_err': col('MG_quick_err'),
        }
    return out


def standardize_metadata(raw, fields=None):
    """Standardize a raw metadata dict -> (features, mask) via the fixed rules.

    ``features`` are NaN-free; ``mask`` is 1 where the raw value was valid. For
    parity with the released latents, extraction passes an all-ones mask (every
    field treated as present); pass the returned mask only when you deliberately
    withhold fields.
    """
    fields = fields or DEFAULT_METADATA_FIELDS
    std = MetadataStandardizer(fields=fields)
    return std.transform(raw, return_mask=True)
