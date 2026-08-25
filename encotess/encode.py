"""High-level light-curve -> latent encoder.

``Encoder`` wraps the trained BiDirectional MinGRU plus the time-aware multi-scale
pooling into a single object. Two entry points:

- ``encode(...)`` — encode in-memory arrays (a single light curve or a batch of
  variable-length curves) to 1536-d latents.
- ``encode_h5(...)`` — batch-encode every light curve in an HDF5 file, reproducing
  the released latent banks bit-for-bit (same trim, metadata, pooling).

``project_pca(...)`` applies the bundled full-rank global PCA (all comps by default).
"""
from __future__ import annotations

import numpy as np
import torch

from encotess.model import BiDirectionalMinGRU
from encotess.metadata import DEFAULT_METADATA_FIELDS, MetadataStandardizer
from encotess.pooling import compute_multiscale_features
from encotess.io import DEFAULT_TRIM_EDGES, read_metadata_table, read_identifiers


def load_encoder(weights_path=None, device=None):
    """Convenience loader for the bundled encoder (see ``Encoder.load``)."""
    return Encoder.load(weights_path=weights_path, device=device)


class Encoder:
    """Trained EncoTESS encoder: light curve -> 1536-d latent."""

    def __init__(self, model: BiDirectionalMinGRU, device: torch.device):
        self.model = model.to(device).eval()
        self.device = device
        self.hidden_size = model.hidden_size
        self._pca = None  # lazily loaded GlobalPCA
        # Locked-in e100 extraction settings — reproduce the released latent banks
        # bit-for-bit. head_norm: apply the trained head LayerNorm to the hidden
        # states before pooling (raw minGRU states are non-negative; without it
        # the latent is uncentered). minmax_edge_skip=0: the e100 banks take the
        # global_min/global_max over the full hidden-state stream (no edge trim);
        # verified to reproduce sendit/e100 latents_pretrain.npz to float32.
        self.apply_head_norm = True
        self.minmax_edge_skip = 0

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, weights_path=None, device=None):
        """Load the encoder checkpoint, auto-detecting architecture from weights."""
        from encotess import assets
        if weights_path is None:
            weights_path = assets.encoder_weights_path()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        sd = (ckpt['model_state_dict']
              if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt)

        has_meta = any(k.startswith('meta_encoder.') for k in sd)
        has_flow = any(k.startswith('flow.') for k in sd)
        has_bwd = any(k.startswith('backward_cell.') for k in sd)
        hidden_size = sd['forward_input_proj.weight'].shape[0]

        num_meta = (ckpt.get('num_meta_features', len(DEFAULT_METADATA_FIELDS))
                    if isinstance(ckpt, dict) else len(DEFAULT_METADATA_FIELDS))
        if not has_meta:
            num_meta = 0
        meta_use_mask = ckpt.get('meta_use_mask', False) if isinstance(ckpt, dict) else False

        model = BiDirectionalMinGRU(
            hidden_size=hidden_size,
            direction='bi' if has_bwd else 'forward',
            mode='parallel',
            use_flow=has_flow,
            num_meta_features=num_meta,
            meta_use_mask=meta_use_mask,
        )
        model.load_state_dict(sd)
        return cls(model, device)

    # --------------------------------------------------------------- helpers
    def _pool(self, h_combined, t_combined):
        """Time-aware multiscale pool -> 1536-d latent (released e100 settings)."""
        return compute_multiscale_features(
            h_combined, t_combined, n_segments=4,
            minmax_edge_skip=self.minmax_edge_skip,
            diff_weight_mode='dt', glob_mode='uniform', seg_mode='equal_count',
            hidden_size=self.hidden_size,
        )

    @torch.no_grad()
    def _forward_batch(self, flux, flux_err, time, lengths, meta_feat, meta_mask):
        """Run one padded batch through the model and pool each curve.

        flux/flux_err/time: (B, Lmax) torch tensors (post-trim, zero-padded).
        lengths: (B,) valid lengths. meta_feat/meta_mask: (B, 13) or None.
        Returns a list of (1536,) numpy latents.
        """
        B, L = flux.shape
        mask = torch.zeros(B, L, device=self.device)
        for i, ln in enumerate(lengths):
            mask[i, :int(ln)] = 1.0
        x_in = torch.stack([flux, flux_err], dim=-1)
        t_in = time.unsqueeze(-1)
        out = self.model(x_in, t_in, mask=mask, metadata=meta_feat,
                         meta_mask=meta_mask, return_states=True)
        h_fwd = out.get('h_fwd_tensor')
        h_bwd = out.get('h_bwd_tensor')
        t_enc = out.get('t_enc')

        # Apply the trained head LayerNorm to [h_fwd ‖ h_bwd ‖ t_enc], then slice
        # off the (normalized) hidden parts — reproducing the released banks.
        if self.apply_head_norm and getattr(self.model, 'head_norm', None) is not None:
            Hd = self.model.hidden_size
            if h_fwd is not None and h_bwd is not None:
                h_bi = torch.cat([h_fwd, h_bwd, t_enc], dim=-1)
            elif h_fwd is not None:
                h_bi = torch.cat([h_fwd, t_enc], dim=-1)
            else:
                h_bi = torch.cat([h_bwd, t_enc], dim=-1)
            h_bi = self.model.head_norm(h_bi)
            if h_fwd is not None and h_bwd is not None:
                h_fwd, h_bwd = h_bi[..., :Hd], h_bi[..., Hd:2 * Hd]
            elif h_fwd is not None:
                h_fwd = h_bi[..., :Hd]
            else:
                h_bwd = h_bi[..., :Hd]

        latents = []
        for i in range(B):
            vl = int(lengths[i])
            if h_fwd is not None and h_bwd is not None:
                h = torch.cat([h_fwd[i, :vl, :], h_bwd[i, :vl, :]], dim=-1)
            elif h_fwd is not None:
                h = h_fwd[i, :vl, :]
            else:
                h = h_bwd[i, :vl, :]
            latents.append(self._pool(h, time[i, :vl]).cpu().numpy())
        return latents

    def _prep_metadata(self, metadata, n, use_mask):
        """Turn a metadata dict / array into (features, mask) tensors."""
        if metadata is None:
            return None, None
        if isinstance(metadata, dict):
            raw = {f: np.atleast_1d(np.asarray(metadata.get(f, np.nan), dtype=np.float32))
                   for f in DEFAULT_METADATA_FIELDS}
            # broadcast scalars to n
            raw = {f: (v if len(v) == n else np.full(n, v[0], dtype=np.float32))
                   for f, v in raw.items()}
            std = MetadataStandardizer(fields=DEFAULT_METADATA_FIELDS)
            feat, mask = std.transform(raw, return_mask=True)
        else:
            feat = np.asarray(metadata, dtype=np.float32)
            mask = np.ones_like(feat)
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device)
        mask_t = (torch.tensor(mask, dtype=torch.float32, device=self.device)
                  if use_mask else None)
        return feat_t, mask_t

    # ---------------------------------------------------- in-memory prep
    def _prepare_inputs(self, flux, flux_err, time, metadata, trim_edges):
        """Trim, pad, and standardize in-memory light curve(s) for the model.

        Shared by ``encode`` and ``flux.predict_flux`` so both apply identical
        trimming/padding/metadata handling. Returns a dict of model-ready tensors
        plus the per-curve trimmed time axes and a ``single`` flag.
        """
        single = np.ndim(flux[0]) == 0 if isinstance(flux, (list, tuple)) else np.ndim(flux) == 1
        if single:
            flux_list = [np.asarray(flux, dtype=np.float32)]
            ferr_list = [np.asarray(flux_err, dtype=np.float32)]
            time_list = [np.asarray(time, dtype=np.float32)]
        else:
            flux_list = [np.asarray(a, dtype=np.float32) for a in flux]
            ferr_list = [np.asarray(a, dtype=np.float32) for a in flux_err]
            time_list = [np.asarray(a, dtype=np.float32) for a in time]

        te = int(trim_edges)
        if te > 0:
            def _trim(a):
                return a[te:len(a) - te] if len(a) > 2 * te else a[:0]
            flux_list = [_trim(a) for a in flux_list]
            ferr_list = [_trim(a) for a in ferr_list]
            time_list = [_trim(a) for a in time_list]

        lengths = np.array([len(a) for a in flux_list])
        if (lengths < 1).any():
            raise ValueError('A light curve is empty after trimming; reduce trim_edges.')
        Lmax = int(lengths.max())
        B = len(flux_list)

        def _pad(lst):
            out = np.zeros((B, Lmax), dtype=np.float32)
            for i, a in enumerate(lst):
                out[i, :len(a)] = a
            return torch.tensor(out, device=self.device)

        meta_feat, meta_mask = self._prep_metadata(metadata, B, self.model.meta_use_mask)
        return {
            'flux': _pad(flux_list), 'flux_err': _pad(ferr_list), 'time': _pad(time_list),
            'lengths': lengths, 'meta_feat': meta_feat, 'meta_mask': meta_mask,
            'time_list': time_list, 'single': single,
        }

    # ---------------------------------------------------------------- encode
    def encode(self, flux, flux_err, time, metadata=None,
               trim_edges=DEFAULT_TRIM_EDGES):
        """Encode in-memory light curve(s) to 1536-d latent(s).

        Args:
            flux, flux_err, time: a single 1-D array each, or a list of 1-D arrays
                (variable length) for a batch.
            metadata: dict of the 13 metadata fields (scalars for a single curve,
                or length-B arrays for a batch), or a pre-standardized (B, 13)
                array, or None. Missing fields may be NaN.
            trim_edges: leading/trailing raw samples to drop (default 10, matches
                training).

        Returns:
            (1536,) for a single curve, or (B, 1536) for a batch.
        """
        p = self._prepare_inputs(flux, flux_err, time, metadata, trim_edges)
        latents = self._forward_batch(p['flux'], p['flux_err'], p['time'],
                                      p['lengths'], p['meta_feat'], p['meta_mask'])
        latents = np.stack(latents, axis=0)
        return latents[0] if p['single'] else latents

    @torch.no_grad()
    def encode_h5(self, h5_path, indices=None, trim_edges=DEFAULT_TRIM_EDGES,
                  batch_size=32, return_identifiers=True, progress=False):
        """Batch-encode an H5 file, reproducing the released latent banks.

        Returns ``latents`` (N, 1536); if ``return_identifiers`` also a dict of
        gaia_ids/tic_ids/sectors/bprp0/... aligned row-for-row.
        """
        import h5py

        idx = None if indices is None else np.asarray(indices)
        with h5py.File(h5_path, 'r') as f:
            lengths = (f['length'][:] if idx is None else f['length'][idx]).astype(np.int64)
        te = int(trim_edges)
        if te > 0:
            lengths = np.maximum(lengths - 2 * te, 0)
        n = len(lengths)

        # Metadata (all fields present -> all-ones mask; matches release).
        raw = read_metadata_table(h5_path, DEFAULT_METADATA_FIELDS, indices=idx)
        std = MetadataStandardizer(fields=DEFAULT_METADATA_FIELDS)
        meta_all = std.transform(raw)               # (n, 13), NaN->0
        meta_mask_all = np.ones_like(meta_all)

        latents = []
        with h5py.File(h5_path, 'r') as f:
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                rows = (np.arange(start, end) if idx is None else idx[start:end])
                lb = lengths[start:end]
                bmax = int(lb.max())
                s0, s1 = te, te + bmax
                flux = torch.tensor(f['flux'][rows, s0:s1], dtype=torch.float32, device=self.device)
                ferr = torch.tensor(f['flux_err'][rows, s0:s1], dtype=torch.float32, device=self.device)
                time = torch.tensor(f['time'][rows, s0:s1], dtype=torch.float32, device=self.device)
                mf = torch.tensor(meta_all[start:end], dtype=torch.float32, device=self.device)
                mm = (torch.tensor(meta_mask_all[start:end], dtype=torch.float32, device=self.device)
                      if self.model.meta_use_mask else None)
                latents.extend(self._forward_batch(flux, ferr, time, lb, mf, mm))
                if progress and (start // batch_size) % 20 == 0:
                    print(f'  encoded {end}/{n}')
        latents = np.stack(latents, axis=0)

        if return_identifiers:
            return latents, read_identifiers(h5_path, indices=idx)
        return latents

    # ------------------------------------------------------------------ pca
    def project_pca(self, latents, dim=None):
        """Project 1536-d latents to the global-PCA basis.

        The bundled PCA is full-rank (1536 comps) and unwhitened, so ``dim=None``
        returns the full lossless PCA re-expression of the latent; pass ``dim=16``
        or ``dim=64`` for a compact encoding (top-64 ~= 97.9% of the variance).
        """
        if self._pca is None:
            from encotess import assets
            from encotess.pca import GlobalPCA
            self._pca = GlobalPCA.from_npz(assets.pca_weights_path())
        z = np.atleast_2d(np.asarray(latents, dtype=np.float64))
        out = self._pca.transform(z, dim=dim)
        return out[0] if np.ndim(latents) == 1 else out
