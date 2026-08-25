"""BiDirectional MinGRU light-curve encoder (EncoTESS core).

A single-cell minGRU run forward and backward over a variable-length light curve,
conditioned per-timestep on a non-linear time encoding and a stellar-metadata
embedding. A zuko Neural Spline Flow head models the per-timestep flux
distribution ``p(flux | context)``.

This is the inference-only, release build: the research model's convolutional
channels, split metadata encoders, and DANN sector adversary have been removed
(the released ``sendit/e50`` checkpoint was trained without any of them, so the
state dict loads unchanged). The log-domain parallel scan and the hard
recurrence gating of masked positions are preserved exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from encotess.metadata import MetadataEncoder

try:
    import zuko
except Exception:  # pragma: no cover - zuko is a hard dependency for the flow head
    zuko = None


class minGRUCell(nn.Module):
    """Minimal GRU cell (single-step) with a log-domain parallel scan."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_z = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(input_size, hidden_size)

    @staticmethod
    def g(x):
        return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

    @staticmethod
    def log_g(x):
        # log g(x) with g(x)=relu(x)+0.5 for x>=0 and sigmoid(x) for x<0.
        # Continuous at 0 (both branches give log(0.5)).
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    @staticmethod
    def parallel_scan_log(log_coeffs, log_values):
        # Two cumulative ops instead of a T-step Python loop → O(1) graph depth.
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h)[:, 1:]

    def step(self, x_t, h_prev=None, mask_t=None):
        if h_prev is None:
            h_prev = x_t.new_zeros(x_t.shape[0], self.W_h.out_features)
        z = torch.sigmoid(self.W_z(x_t))
        h_tilde = torch.tanh(self.W_h(x_t))
        h = (1.0 - z) * h_prev + z * h_tilde
        if mask_t is not None:
            # Hard recurrence gating: gated steps (mask_t == 0) carry the state.
            keep = (mask_t > 0.5).unsqueeze(-1)
            h = torch.where(keep, h, h_prev)
        return h

    def step_parallel(self, x, h_0, mask=None):
        # x: (B, T, input_size); h_0: (B, 1, hidden_size); mask: (B, T) optional.
        k = self.W_z(x)
        log_z = -F.softplus(-k)                    # log sigmoid(k)
        log_coeffs = -F.softplus(k)                # log(1 - sigmoid(k))
        if mask is not None:
            keep = (mask > 0.5).unsqueeze(-1)      # (B, T, 1)
            log_z = torch.where(keep, log_z, log_z.new_full((), -1e9))
            log_coeffs = torch.where(keep, log_coeffs, log_coeffs.new_zeros(()))
        log_h_0 = h_0.clamp_min(torch.finfo(h_0.dtype).tiny).log()
        log_h_tilde = minGRUCell.log_g(self.W_h(x))
        return minGRUCell.parallel_scan_log(
            log_coeffs, torch.cat([log_h_0, log_z + log_h_tilde], dim=1)
        )


class BiDirectionalMinGRU(nn.Module):
    """Bidirectional minGRU encoder with a metadata path and a flow flux head."""

    def __init__(
        self,
        hidden_size: int = 64,
        direction: str = "bi",
        mode: str = "parallel",
        use_flow: bool = False,
        num_meta_features: int = 13,
        meta_dropout: float = 0.1,
        meta_use_mask: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_flow = use_flow
        self.direction = direction
        self.mode = mode
        self.meta_use_mask = meta_use_mask

        num_time_enc_dims = 8
        self.num_time_enc_dims = num_time_enc_dims
        self.time_enc = nn.Sequential(
            nn.Linear(1, num_time_enc_dims),
            nn.ReLU(),
            nn.Linear(num_time_enc_dims, num_time_enc_dims),
        )

        # Stellar-metadata encoder: 3×128 hidden, 32-d embedding broadcast to
        # every RNN timestep.
        self.num_meta_features = num_meta_features
        if num_meta_features > 0:
            meta_emb_dim = 32
            self.meta_encoder = MetadataEncoder(
                input_dim=num_meta_features,
                latent_dim=meta_emb_dim,
                hidden_dims=[128, 128, 128],
                dropout=meta_dropout,
                use_mask=meta_use_mask,
            )
        else:
            meta_emb_dim = 0
            self.meta_encoder = None
        self.meta_emb_dim = meta_emb_dim

        rnn_input_dim = 2 + num_time_enc_dims + meta_emb_dim
        self.forward_input_proj = nn.Linear(rnn_input_dim, hidden_size)
        self.backward_input_proj = nn.Linear(rnn_input_dim, hidden_size)

        self.forward_cell = minGRUCell(hidden_size, hidden_size)
        self.backward_cell = minGRUCell(hidden_size, hidden_size)

        if self.direction == 'bi':
            self.output_size = hidden_size * 2 + num_time_enc_dims
        else:
            self.output_size = hidden_size + num_time_enc_dims
        self.head_norm = nn.LayerNorm(self.output_size)
        self.time_scale = nn.Parameter(torch.tensor(5.0))

        head_hidden = max(32, hidden_size // 2)
        self.gauss_head = nn.Sequential(
            nn.Linear(self.output_size, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

        self.flow = None
        if zuko is not None and use_flow:
            flow_context_dim = self.output_size + 1  # + measurement error
            self.flow = zuko.flows.NSF(1, flow_context_dim, transforms=2,
                                       hidden_features=[64, 64])

    def forward(self, x, t, mask=None, metadata=None, meta_mask=None,
                meta_block_drop=None, return_states: bool = False,
                flow_mode: str = 'mean'):
        """Encode a batch of light curves.

        Args:
            x: (B, L, 2) [flux, flux_err].
            t: (B, L) or (B, L, 1) timestamps.
            mask: (B, L) 1=observed, 0=masked/padded (gated out of the recurrence).
            metadata: (B, num_meta_features) standardized metadata, or None.
            meta_mask: (B, num_meta_features) validity mask (used when
                ``meta_use_mask=True``).
            meta_block_drop: (B,) bool; where True the metadata embedding is zeroed.
            return_states: if True, return per-timestep hidden states + time
                encoding (for pooling) instead of the reconstructed flux.
            flow_mode: point-estimate mode for the flux head ('mean'|'mode'|'sample').

        Returns:
            dict. With ``return_states=False``: ``{'reconstructed': (B, L, 1)}``.
            With ``return_states=True``: hidden-state tensors + ``t_enc``.
        """
        if x.dim() != 3 or x.size(-1) != 2:
            raise ValueError(f"Expected (B, L, 2), got {tuple(x.shape)}")

        B, L, _ = x.shape

        # Preserve pre-mask flux_err so the flow head always conditions on the
        # real measurement error, even at masked positions.
        flux_err_unmasked = x[..., 1].clone()

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        t_seq = t
        if t_seq.dim() == 3 and t_seq.size(-1) == 1:
            t_seq = t_seq.squeeze(-1)
        if t_seq.dim() != 2:
            raise ValueError("t must be shape (B, L) or (B, L, 1)")

        t0 = t_seq[:, 0].unsqueeze(1)
        t_shifted = t_seq - t0
        t_enc = self.time_enc(t_shifted.unsqueeze(-1))  # (B, L, Te)

        if self.meta_encoder is not None and metadata is not None:
            # MetadataEncoder ignores `mask` when built with use_mask=False.
            meta_emb = self.meta_encoder(metadata, meta_mask)  # (B, meta_emb_dim)
            if meta_block_drop is not None:
                keep = (~meta_block_drop).to(meta_emb.dtype).unsqueeze(-1)
                meta_emb = meta_emb * keep
            meta_emb_seq = meta_emb.unsqueeze(1).expand(-1, L, -1)
        else:
            meta_emb_seq = None

        x_parts = [x, t_enc]
        if meta_emb_seq is not None:
            x_parts.append(meta_emb_seq)
        x = torch.cat(x_parts, dim=-1)

        # ---- Backward scan (recurrence gated on `mask`) ----
        if self.direction in ['bi', 'backward']:
            h_bwd_tensor = x.new_zeros(B, L, self.hidden_size)
            h_bwd = x.new_zeros(B, self.hidden_size)
            mask_bwd = mask.flip(dims=[1]) if mask is not None else None

            if self.mode == 'parallel':
                inp_bwd = x.flip(dims=[1])
                x_bwd_proj = self.backward_input_proj(inp_bwd)
                h_bwd_all = self.backward_cell.step_parallel(
                    x_bwd_proj, h_bwd.unsqueeze(1), mask=mask_bwd)
                h_bwd_tensor = h_bwd_all.flip(dims=[1])
                h0_b = h_bwd_tensor.new_zeros(B, 1, self.hidden_size)
                h_bwd_tensor = torch.cat([h_bwd_tensor[:, 1:, :], h0_b], dim=1)
            elif self.mode == 'sequential':
                for ti in reversed(range(L)):
                    h_bwd_tensor[:, ti, :] = h_bwd
                    inp_bwd = self.backward_input_proj(x[:, ti, :])
                    mask_ti = mask[:, ti] if mask is not None else None
                    h_bwd = self.backward_cell.step(inp_bwd, h_bwd, mask_t=mask_ti)

        # ---- Forward scan ----
        if self.direction in ['bi', 'forward']:
            h_fwd_tensor = x.new_zeros(B, L, self.hidden_size)
            h_fwd = x.new_zeros(B, self.hidden_size)

            if self.mode == 'parallel':
                x_fwd_proj = self.forward_input_proj(x)
                h_fwd_all = self.forward_cell.step_parallel(
                    x_fwd_proj, h_fwd.unsqueeze(1), mask=mask)
                h_fwd_tensor = h_fwd_all
                h0_f = h_fwd_tensor.new_zeros(B, 1, self.hidden_size)
                h_fwd_tensor = torch.cat([h0_f, h_fwd_tensor[:, :-1, :]], dim=1)
            elif self.mode == 'sequential':
                for ti in range(L):
                    h_fwd_tensor[:, ti, :] = h_fwd
                    inp_fwd = self.forward_input_proj(x[:, ti, :])
                    mask_ti = mask[:, ti] if mask is not None else None
                    h_fwd = self.forward_cell.step(inp_fwd, h_fwd, mask_t=mask_ti)

        out = {'reconstructed': None}
        if not return_states:
            seq_prediction = []
            for ti in range(L):
                time_enc_t = t_enc[:, ti, :]
                if self.direction == 'forward':
                    h_bi = torch.cat([h_fwd_tensor[:, ti, :], time_enc_t], dim=1)
                elif self.direction == 'backward':
                    h_bi = torch.cat([h_bwd_tensor[:, ti, :], time_enc_t], dim=1)
                else:
                    h_bi = torch.cat([h_fwd_tensor[:, ti, :],
                                      h_bwd_tensor[:, ti, :], time_enc_t], dim=1)
                h_bi = self.head_norm(h_bi)
                nt = self.num_time_enc_dims
                if nt > 0:
                    h_hidden = h_bi[:, :-nt]
                    h_time = h_bi[:, -nt:] * self.time_scale
                    h_bi = torch.cat([h_hidden, h_time], dim=1)
                meas_err_t = flux_err_unmasked[:, ti]
                if self.flow is not None:
                    ctx = torch.cat([h_bi, meas_err_t.unsqueeze(1)], dim=1)
                    dist = self.flow(ctx)
                    if flow_mode == 'sample':
                        point_prediction = dist.sample().view(B, 1)
                    elif flow_mode == 'mode':
                        ctx_detached = ctx.detach()
                        y_opt = self.flow(ctx_detached).sample().clone().detach()
                        y_opt = y_opt.requires_grad_(True)
                        optimizer = torch.optim.Adam([y_opt], lr=0.1)
                        for _ in range(30):
                            optimizer.zero_grad()
                            loss = -self.flow(ctx_detached).log_prob(y_opt).sum()
                            loss.backward()
                            optimizer.step()
                            with torch.no_grad():
                                y_opt.clamp_(-10, 10)
                        point_prediction = y_opt.detach().view(B, 1)
                    else:  # 'mean'
                        n_samples = 16
                        s_acc = dist.sample()
                        for _ in range(n_samples - 1):
                            s_acc = s_acc + dist.sample()
                        point_prediction = (s_acc / n_samples).view(B, 1)
                else:
                    point_prediction = self.gauss_head(h_bi)
                seq_prediction.append(point_prediction)
            out['reconstructed'] = torch.stack(seq_prediction, dim=1)  # (B, L, 1)

        if return_states:
            if self.direction in ['bi', 'forward']:
                out['h_fwd_tensor'] = h_fwd_tensor
            if self.direction in ['bi', 'backward']:
                out['h_bwd_tensor'] = h_bwd_tensor
            out['t_enc'] = t_enc
        return out
