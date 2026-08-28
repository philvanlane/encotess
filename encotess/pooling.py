"""Time-aware multi-scale pooling: per-timestep hidden states -> one latent vector.

The function defaults (``glob_mode='uniform'``, ``seg_mode='equal_count'``,
``diff_weight_mode='dt'``, ``n_segments=4``) are the pooling configuration used to
produce the released encodings, so a bidirectional hidden state of width H=128
yields a 12*H = 1536-d latent:

    [glob_mean, glob_std, glob_max, glob_min,
     seg0..seg3, first_h, last_h, diff_mean, diff_std]

Pool operators are defined in physical time (Voronoi/segment weighting, rate
Δh/Δt for the difference blocks), not sample index, so irregular cadence and
variable sector length don't bias the latent.
"""
from __future__ import annotations

import torch


def compute_multiscale_features(h_valid: torch.Tensor, t_valid: torch.Tensor,
                                n_segments: int = 4,
                                minmax_edge_skip: int = 0,
                                diff_weight_mode: str = 'dt',
                                minmax_quantile: float = 0.0,
                                subtract_temporal_mean: bool = False,
                                glob_mode: str = 'uniform',
                                seg_mode: str = 'equal_count',
                                hidden_size: int = None,
                                voronoi_cap_factor: float = 3.0) -> torch.Tensor:
    """Compute multi-scale temporal features from hidden states (time-aware).

    Args:
        h_valid: (L, H) hidden states for valid timesteps.
        t_valid: (L,) monotonic physical time per hidden state (e.g. days).
        n_segments: number of equal-time / equal-count segments.
        (other args): see the module docstring; defaults reproduce the released
            latents.

    Returns:
        Feature vector of shape (12*H,).
    """
    L, H = h_valid.shape
    device, dtype = h_valid.device, h_valid.dtype
    eps = torch.finfo(dtype).eps
    features = []

    # Per-sample weights for the time-average blocks.
    if glob_mode == 'uniform' or L == 1:
        w = torch.ones(L, device=device, dtype=dtype)
    elif glob_mode in ('voronoi', 'voronoi_capped'):
        dt = (t_valid[1:] - t_valid[:-1]).to(dtype).clamp_min(eps)
        if glob_mode == 'voronoi_capped':
            cap = (voronoi_cap_factor * dt.median()).clamp_min(eps)
            dt = dt.clamp_max(cap)
        w = torch.empty(L, device=device, dtype=dtype)
        w[0] = 0.5 * dt[0]
        w[-1] = 0.5 * dt[-1]
        if L > 2:
            w[1:-1] = 0.5 * (dt[1:] + dt[:-1])
        w = w.clamp_min(0.0)
    else:
        raise ValueError(f"Unknown glob_mode: {glob_mode!r}")
    w_sum = w.sum().clamp_min(eps)
    w_col = w.unsqueeze(1)

    # 1. Global statistics — time-weighted.
    global_mean = (w_col * h_valid).sum(dim=0) / w_sum
    centered = h_valid - global_mean.unsqueeze(0)
    global_var = (w_col * centered.pow(2)).sum(dim=0) / w_sum
    global_std = global_var.clamp_min(0.0).sqrt()

    if subtract_temporal_mean:
        h_pool = centered
        global_mean = torch.zeros_like(global_mean)
    else:
        h_pool = h_valid

    m = max(0, int(minmax_edge_skip))
    if m > 0 and L > 2 * m + 1:
        h_inner = h_pool[m:L - m, :]
    else:
        h_inner = h_pool

    q = float(minmax_quantile)
    if q > 0.0:
        qs = torch.tensor([q, 1.0 - q], device=device, dtype=dtype)
        quantiles = torch.quantile(h_inner, qs, dim=0)
        global_min = quantiles[0]
        global_max = quantiles[1]
    else:
        global_max = h_inner.max(dim=0).values
        global_min = h_inner.min(dim=0).values
    features.extend([global_mean, global_std, global_max, global_min])

    # 2. Segment pooling.
    if seg_mode in ('equal_time', 'equal_time_carry'):
        t_min = t_valid[0]
        t_max = t_valid[-1]
        span = (t_max - t_min).clamp_min(eps)
        edges = t_min + span * torch.linspace(0.0, 1.0, n_segments + 1,
                                              device=device, dtype=t_valid.dtype)
        Hd = hidden_size if (hidden_size is not None and 2 * hidden_size == H) else None
        for seg_idx in range(n_segments):
            lo = edges[seg_idx]
            hi = edges[seg_idx + 1]
            in_bin = (t_valid >= lo) & (t_valid < hi) if seg_idx < n_segments - 1 \
                     else (t_valid >= lo) & (t_valid <= hi)
            if in_bin.any():
                ws = w[in_bin]
                denom = ws.sum().clamp_min(eps)
                seg_mean = (ws.unsqueeze(1) * h_pool[in_bin, :]).sum(dim=0) / denom
            elif seg_mode == 'equal_time_carry':
                a = int((t_valid < lo).sum().item()) - 1
                a = min(max(a, 0), L - 1)
                b = min(a + 1, L - 1)
                if Hd is not None:
                    seg_mean = torch.cat([h_pool[a, :Hd], h_pool[b, Hd:2 * Hd]], dim=0)
                else:
                    seg_mean = h_pool[a, :]
            else:
                seg_mean = torch.zeros(H, device=device, dtype=dtype)
            features.append(seg_mean)
    elif seg_mode == 'equal_count':
        idx_edges = torch.linspace(0, L, n_segments + 1).round().to(torch.long)
        for seg_idx in range(n_segments):
            s = int(idx_edges[seg_idx]); e = int(idx_edges[seg_idx + 1])
            if e > s:
                ws = w[s:e]
                denom = ws.sum().clamp_min(eps)
                seg_mean = (ws.unsqueeze(1) * h_pool[s:e, :]).sum(dim=0) / denom
            else:
                seg_mean = torch.zeros(H, device=device, dtype=dtype)
            features.append(seg_mean)
    else:
        raise ValueError(f"Unknown seg_mode: {seg_mode!r}")

    # 3. First / last hidden states.
    features.append(h_pool[0, :])
    features.append(h_pool[-1, :])

    # 4. Temporal-derivative statistics — rate Δh/Δt.
    if L > 1:
        dt_step = (t_valid[1:] - t_valid[:-1]).to(dtype).clamp_min(eps)
        rates = (h_valid[1:, :] - h_valid[:-1, :]) / dt_step.unsqueeze(1)
        if diff_weight_mode == 'dt':
            w_d = dt_step.unsqueeze(1)
            w_d_sum = w_d.sum().clamp_min(eps)
            diff_mean = (w_d * rates).sum(dim=0) / w_d_sum
            if rates.shape[0] > 1:
                centered = rates - diff_mean.unsqueeze(0)
                diff_var = (w_d * centered.pow(2)).sum(dim=0) / w_d_sum
                diff_std = diff_var.clamp_min(0.0).sqrt()
            else:
                diff_std = torch.zeros(H, device=device, dtype=dtype)
        elif diff_weight_mode == 'dt_inverse':
            w_d = (1.0 / dt_step).unsqueeze(1)
            w_d_sum = w_d.sum().clamp_min(eps)
            diff_mean = (w_d * rates).sum(dim=0) / w_d_sum
            if rates.shape[0] > 1:
                centered = rates - diff_mean.unsqueeze(0)
                diff_var = (w_d * centered.pow(2)).sum(dim=0) / w_d_sum
                diff_std = diff_var.clamp_min(0.0).sqrt()
            else:
                diff_std = torch.zeros(H, device=device, dtype=dtype)
        elif diff_weight_mode == 'unweighted':
            diff_mean = rates.mean(dim=0)
            diff_std = rates.std(dim=0) if rates.shape[0] > 1 \
                else torch.zeros(H, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown diff_weight_mode: {diff_weight_mode!r}")
        features.extend([diff_mean, diff_std])
    else:
        features.extend([torch.zeros(H, device=device, dtype=dtype),
                         torch.zeros(H, device=device, dtype=dtype)])

    return torch.cat(features, dim=0)
