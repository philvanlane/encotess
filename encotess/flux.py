"""Flux prediction from a light curve (offset-k flow forecast with uncertainty).

The encoder's flow head is trained with a bounded-horizon future objective: from
the bidirectional hidden state it predicts the flux a few timesteps away. This
module reproduces that prediction — at each target position ``j`` it conditions on
the forward state ``k`` steps earlier and the backward state ``k`` steps later,
then draws from the flow to get a median prediction and a p16–p84 band.

This mirrors the research reconstruction (``compute_flow_predictions``); the band
is the model's genuine predictive uncertainty at horizon ``k``.
"""
from __future__ import annotations

import numpy as np
import torch

from encotess.io import DEFAULT_TRIM_EDGES


@torch.no_grad()
def predict_flux(encoder, flux, flux_err, time, metadata=None,
                 trim_edges=DEFAULT_TRIM_EDGES, offset=8, n_samples=64,
                 mask=None, return_time=True):
    """Predict flux at horizon ``offset`` for a single light curve.

    Args:
        encoder: a loaded ``encotess.Encoder`` (must have a flow head).
        flux, flux_err, time: 1-D arrays for one light curve.
        metadata: metadata dict / standardized (1, 13) array / None.
        trim_edges: leading/trailing raw samples dropped first (default 10).
        offset: prediction horizon k in timesteps (default 8; needs 2k < L).
        n_samples: flow samples used to form the median/percentile band.
        mask: optional (L,) array (1=observed, 0=masked); targets in masked
            regions pull their source states from the nearest observed sample.
        return_time: also return the target time axis.

    Returns:
        dict with 'flux' (median), 'p16', 'p84' over the target positions, and
        (if ``return_time``) 'time'. Predictions are in the model's flux units.
    """
    model = encoder.model
    device = encoder.device
    if model.flow is None:
        raise RuntimeError('The loaded encoder has no flow head; cannot predict flux.')

    flux = np.asarray(flux, dtype=np.float32)
    flux_err = np.asarray(flux_err, dtype=np.float32)
    time = np.asarray(time, dtype=np.float32)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32)

    te = int(trim_edges)
    if te > 0 and len(flux) > 2 * te:
        sl = slice(te, len(flux) - te)
        flux, flux_err, time = flux[sl], flux_err[sl], time[sl]
        if mask is not None:
            mask = mask[sl]

    L = len(flux)
    k = int(offset)
    if 2 * k >= L:
        raise ValueError(f'offset {k} too large for length {L} (need 2k < L).')

    flux_t = torch.tensor(flux, device=device).unsqueeze(0)
    ferr_t = torch.tensor(flux_err, device=device).unsqueeze(0)
    time_t = torch.tensor(time, device=device).unsqueeze(0)
    mask_t = (torch.ones(1, L, device=device) if mask is None
              else torch.tensor(mask, device=device).unsqueeze(0))
    meta_feat, meta_mask = encoder._prep_metadata(metadata, 1, model.meta_use_mask)

    out = model(torch.stack([flux_t, ferr_t], dim=-1), time_t.unsqueeze(-1),
                mask=mask_t, metadata=meta_feat, meta_mask=meta_mask,
                return_states=True)
    h_fwd, h_bwd, t_enc = out.get('h_fwd_tensor'), out.get('h_bwd_tensor'), out['t_enc']
    Te = t_enc.size(-1)
    n_targets = L - 2 * k
    t_tgt = t_enc[:, k:L - k, :]

    # Nearest-unmasked source remap (identity when fully observed).
    idx_row = torch.arange(L, device=device).unsqueeze(0)
    fwd_remap, _ = torch.cummax(
        torch.where(mask_t > 0.5, idx_row, torch.full_like(idx_row, -1)), dim=1)
    fwd_remap = fwd_remap.clamp(min=0)
    bwd_flipped, _ = torch.cummin(
        torch.where(mask_t > 0.5, idx_row, torch.full_like(idx_row, L)).flip(dims=[1]), dim=1)
    bwd_remap = bwd_flipped.flip(dims=[1]).clamp(max=L - 1)
    fwd_idx = fwd_remap[:, :n_targets].long()
    bwd_idx = bwd_remap[:, 2 * k:].long()

    if model.direction == 'bi' and h_fwd is not None and h_bwd is not None:
        H = h_fwd.size(-1)
        src_f = torch.gather(h_fwd, 1, fwd_idx.unsqueeze(-1).expand(-1, -1, H))
        src_b = torch.gather(h_bwd, 1, bwd_idx.unsqueeze(-1).expand(-1, -1, H))
        flat_in = torch.cat([src_f, src_b, t_tgt], dim=-1).reshape(-1, 2 * H + Te)
    else:
        h = h_fwd if h_fwd is not None else h_bwd
        H = h.size(-1)
        src = torch.gather(h, 1, fwd_idx.unsqueeze(-1).expand(-1, -1, H))
        flat_in = torch.cat([src, t_tgt], dim=-1).reshape(-1, H + Te)

    normed = model.head_norm(flat_in)
    if Te > 0:
        normed = torch.cat([normed[:, :-Te], normed[:, -Te:] * model.time_scale], dim=1)
    ferr_flat = ferr_t[:, k:L - k].reshape(-1, 1)
    dist = model.flow(torch.cat([normed, ferr_flat], dim=1))
    samples = torch.stack([dist.sample() for _ in range(n_samples)], dim=0).squeeze(-1)
    median = torch.quantile(samples, 0.5, dim=0).cpu().numpy()
    p16 = torch.quantile(samples, 0.16, dim=0).cpu().numpy()
    p84 = torch.quantile(samples, 0.84, dim=0).cpu().numpy()

    result = {'flux': median, 'p16': p16, 'p84': p84}
    if return_time:
        result['time'] = time[k:L - k]
    return result
