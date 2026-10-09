"""Bayesian age inference from the PLS-3 projection of a latent.

The bundled model is a **neural likelihood estimator**, not a regressor. A
normalizing flow (zuko NSF) learns the conditional density

    p(PLS-3 features | log10(age/Myr), BPRP0, log10 BPRP0_err)

and an age is recovered by inverting it on a grid, against a uniform prior over
``log10(age/Myr)``:

    p(age | features, colour) ∝ p(features | age, colour) · p(age)

So the output is a **posterior**, not a point estimate; ``infer`` returns its
median and 16th/84th percentiles, and can return the full grid.

Conditioning on colour is deliberate: a latent's age signal is entangled with
spectral type, and supplying BPRP0 (dereddened) lets the flow model the age
dependence at fixed colour. The likelihood is mixed with a uniform outlier
component over the feature box, so a star landing far outside the training
distribution degrades gracefully instead of producing a confident wrong answer.

Usage::

    import encotess
    enc  = encotess.load_encoder()
    age  = encotess.load_age_model()
    t3   = enc.project_pls(star_latent, dim=3)      # star-aggregated latent
    post = age.infer(t3, bprp0=1.1, bprp0_err=0.03)
    print(post['median'], post['p16'], post['p84'])  # log10(age/Myr)

Scope and caveats
-----------------
This model was trained on **all** 9,413 labelled stars with no held-out split —
only a 15% carve-out to pick the stopping epoch. It is therefore a demonstration
of the inference machinery, **not** a validated age predictor, and any accuracy
measured on a star it was trained on is in-sample and optimistic. Honest
held-out accuracy requires the cross-validated pipeline, where both the PLS
projection and the flow are refit inside every fold.

The PLS projection feeding this model is itself supervised (see
``encotess.pls``), which is a second reason not to read held-out performance
into numbers produced here.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def load_age_model(weights_path=None, device=None):
    """Convenience loader for the bundled age model (see ``AgeNLE.load``)."""
    return AgeNLE.load(weights_path=weights_path, device=device)


class _NLEFlow(nn.Module):
    """Conditional NSF over the observable, context = (log_age, bprp0, log_bprp0_err)."""

    def __init__(self, features, context=3, transforms=6, hidden_features=(64, 64)):
        super().__init__()
        import zuko
        self.flow = zuko.flows.NSF(features=features, context=context,
                                   transforms=transforms,
                                   hidden_features=list(hidden_features))


class AgeNLE:
    """Bundled PLS-3 age likelihood model with grid-Bayes inversion."""

    def __init__(self, flow, feat_mean, feat_std, prior_loga_myr,
                 pls_dim=3, n_train_stars=None, device=None):
        self.flow = flow
        self.feat_mean = np.asarray(feat_mean, dtype=np.float64)
        self.feat_std = np.asarray(feat_std, dtype=np.float64)
        self.prior_loga_myr = tuple(float(v) for v in prior_loga_myr)
        self.pls_dim = int(pls_dim)
        self.n_train_stars = None if n_train_stars is None else int(n_train_stars)
        self.device = device or torch.device('cpu')

    @classmethod
    def load(cls, weights_path=None, device=None):
        """Load the bundled checkpoint (CPU by default)."""
        from encotess import assets
        if weights_path is None:
            weights_path = assets.age_weights_path()
        if device is None:
            device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        module = _NLEFlow(features=ckpt['features'],
                          context=ckpt.get('context', 3),
                          transforms=ckpt['flow_transforms'],
                          hidden_features=ckpt['flow_hidden_features'])
        missing = module.load_state_dict(
            {k: v for k, v in ckpt['state_dict'].items() if not k.startswith('ln_p_outlier')},
            strict=False)
        del missing
        module.to(device).eval()
        return cls(module, ckpt['feat_mean'], ckpt['feat_std'],
                   ckpt['prior_loga_myr'], pls_dim=ckpt.get('pls_dim', 3),
                   n_train_stars=ckpt.get('n_train_stars'), device=device)

    # ------------------------------------------------------------------ infer
    @torch.no_grad()
    def infer(self, features, bprp0, bprp0_err, grid_size: int = 1000,
              return_posterior: bool = False):
        """Infer the age posterior for one star or a batch.

        Args:
            features: (3,) or (N, 3) PLS-3 scores, raw — from
                ``Encoder.project_pls(z, dim=3)`` on a star-aggregated latent.
                Standardization is applied internally.
            bprp0: dereddened Gaia BP-RP colour (scalar or (N,)).
            bprp0_err: its uncertainty (scalar or (N,)); must be positive.
            grid_size: points in the log-age grid spanning ``prior_loga_myr``.
            return_posterior: also return the grid and the posterior density.

        Returns:
            dict of ``median``, ``p16``, ``p84``, ``mean``, ``map`` in
            **log10(age/Myr)**, plus ``age_myr`` (median, linear). Scalars for a
            single star, arrays for a batch. With ``return_posterior``, also
            ``loga_grid`` and ``posterior``.
        """
        X = np.atleast_2d(np.asarray(features, dtype=np.float64))
        single = np.ndim(features) == 1
        if X.shape[1] != self.pls_dim:
            raise ValueError(f'Expected {self.pls_dim} PLS features, got {X.shape[1]}.')
        b = np.atleast_1d(np.asarray(bprp0, dtype=np.float64))
        be = np.atleast_1d(np.asarray(bprp0_err, dtype=np.float64))
        if len(b) == 1 and len(X) > 1:
            b = np.repeat(b, len(X))
        if len(be) == 1 and len(X) > 1:
            be = np.repeat(be, len(X))
        if not (len(b) == len(be) == len(X)):
            raise ValueError('features, bprp0 and bprp0_err must have matching lengths.')
        if not np.isfinite(X).all():
            raise ValueError('features contain non-finite values.')
        if not (np.isfinite(b).all() and np.isfinite(be).all()):
            raise ValueError('bprp0 / bprp0_err contain non-finite values.')

        Xs = (X - self.feat_mean) / self.feat_std
        log_be = np.log10(np.maximum(be, 1e-6))

        dev = self.device
        B, G = len(Xs), int(grid_size)
        grid = torch.linspace(self.prior_loga_myr[0], self.prior_loga_myr[1], G, device=dev)
        z = torch.tensor(Xs, dtype=torch.float32, device=dev)
        bt = torch.tensor(b, dtype=torch.float32, device=dev)
        bet = torch.tensor(log_be, dtype=torch.float32, device=dev)

        z_e = z.unsqueeze(1).expand(B, G, z.shape[1]).reshape(B * G, z.shape[1])
        b_e = bt.unsqueeze(1).expand(B, G).reshape(B * G)
        be_e = bet.unsqueeze(1).expand(B, G).reshape(B * G)
        a_e = grid.unsqueeze(0).expand(B, G).reshape(B * G)
        ctx = torch.stack([a_e, b_e, be_e], dim=1)

        # uniform prior over the grid -> posterior is the normalized likelihood
        log_lik = self.flow.flow(ctx).log_prob(z_e).reshape(B, G)
        post = (log_lik - torch.logsumexp(log_lik, dim=1, keepdim=True)).exp()
        cdf = post.cumsum(dim=1)

        def pct(p):
            return grid[(cdf >= p).float().argmax(dim=1)].cpu().numpy()

        g = grid.cpu().numpy()
        out = {
            'median': pct(0.5), 'p16': pct(0.16), 'p84': pct(0.84),
            'mean': (post * grid.unsqueeze(0)).sum(1).cpu().numpy(),
            'map': g[post.argmax(1).cpu().numpy()],
        }
        out['age_myr'] = 10.0 ** out['median']
        if return_posterior:
            out['loga_grid'] = g
            out['posterior'] = post.cpu().numpy()
        if single:
            out = {k: (v[0] if k != 'loga_grid' and np.ndim(v) else v)
                   for k, v in out.items()}
        return out
