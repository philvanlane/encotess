"""Supervised PLS projection of the 1536-d latents (numpy only, no sklearn).

Where ``GlobalPCA`` rotates the latent to maximise *variance*, this projects it
onto the directions that maximise *covariance with stellar age*. The bundled
artifact reproduces the released ``encodings_pls16_star.npz`` view.

The transform is affine — a per-feature standardization followed by a matrix
multiply::

    X_std = (X - x_mean) / x_std              # centre AND scale each latent dim
    T     = X_std @ x_rotations[:, :dim]      # (N, dim) component scores

Both steps come from ``sklearn.cross_decomposition.PLSRegression(scale=True)``
fit against ``log10(age/Myr)``. Three things differ from the PCA path:

- **It is supervised.** The directions were chosen using the ages of the fit
  stars, so the projection is not independent of the target (see below).
- **``x_rotations`` is not orthonormal.** This is a projection, not a rotation:
  it is not invertible back to the latent, and there is no "explained variance"
  to report per component.
- **Components are nested.** The first ``k`` columns of the 16-component fit are
  exactly a ``k``-component fit (verified to ~1e-16), so ``dim`` just slices.

Scores are returned raw, on the scale of the released encodings (component 1 has
a standard deviation of ~22, not 1). The age pipeline z-scores them per fold
before feeding its flow; that rescaling is deliberately *not* applied here,
because it is a property of a fold, not of the projection.

Rows are **per star**, not per light curve: a star's sector latents are combined
with an element-wise maximum (``aggregate_by_star``) before projecting.

What the bundled projection was fit on: the 9,221 stars flagged ``in_age_fit`` in
the released encoding — the FGKMcal stars carrying an age — against
``log10(age/Myr)``. No hosts or thick-disk stars were in the fit.

To keep fewer than 16 directions, pass ``dim=`` — components are nested, so the
top ``k`` of this fit are what a ``k``-component fit would give.

Leakage
-------
The bundled projection was fit **once, globally**, on the labelled stars. It is a
descriptive view of age structure in the latent space, and it is not a validated
age predictor. Because it saw those stars' ages when choosing its directions, do
not use these scores as input features for an age model that you then evaluate on
any star in the fit set — the evaluation would be optimistic. The released
``encodings_pls16_star.npz`` marks the fit population in its ``in_age_fit``
column; filter on it. A leak-free pipeline refits PLS inside every CV fold on
that fold's training split alone.

Note that refitting per fold controls *target* leakage only. It does nothing
about sector structure, which lives in the latents themselves and is present in
every fold, so a score for a single light curve still depends on which sector
observed it.
"""
from __future__ import annotations

import numpy as np


def aggregate_by_star(latents, gaia_ids, how: str = 'max'):
    """Combine a star's per-light-curve latents into one row per star.

    The released PLS view uses ``how='max'`` (element-wise maximum over the
    star's sectors), so that is the default; match it to reproduce the shipped
    scores.

    Args:
        latents: (N, n_features) per-light-curve latents.
        gaia_ids: (N,) Gaia DR3 identifiers, one per row.
        how: 'max', 'mean' or 'median'.

    Returns:
        (unique_gaia_ids, Z) with Z of shape (n_stars, n_features), ordered by
        ``np.unique`` of the identifiers.
    """
    reducers = {'max': lambda a: a.max(axis=0),
                'mean': lambda a: a.mean(axis=0),
                'median': lambda a: np.median(a, axis=0)}
    if how not in reducers:
        raise ValueError(f"how must be one of {sorted(reducers)}, got {how!r}")
    latents = np.asarray(latents)
    gaia_ids = np.asarray(gaia_ids)
    if len(latents) != len(gaia_ids):
        raise ValueError('latents and gaia_ids must have the same length.')

    reduce = reducers[how]
    unique = np.unique(gaia_ids)
    Z = np.stack([reduce(latents[gaia_ids == g]) for g in unique])
    return unique, Z


class GlobalPLS:
    """Apply the pre-fit supervised PLS projection to latents."""

    def __init__(self, x_mean, x_std, x_rotations, n_fit_stars=None,
                 aggregation='max', target='log10(age_Myr)'):
        self.x_mean = np.asarray(x_mean, dtype=np.float64)
        self.x_std = np.asarray(x_std, dtype=np.float64)
        self.x_rotations = np.asarray(x_rotations, dtype=np.float64)  # (n_feat, k)
        self.n_fit_stars = None if n_fit_stars is None else int(n_fit_stars)
        self.aggregation = str(aggregation)
        self.target = str(target)

    @classmethod
    def from_npz(cls, path):
        """Load from the bundled ``pls_projection.npz`` artifact."""
        z = np.load(path, allow_pickle=False)
        return cls(
            x_mean=z['x_mean'],
            x_std=z['x_std'],
            x_rotations=z['x_rotations'],
            n_fit_stars=z['n_fit_stars'] if 'n_fit_stars' in z.files else None,
            aggregation=str(z['aggregation']) if 'aggregation' in z.files else 'max',
            target=str(z['target']) if 'target' in z.files else 'log10(age_Myr)',
        )

    @property
    def n_components(self) -> int:
        return self.x_rotations.shape[1]

    def transform(self, X, dim: int = None) -> np.ndarray:
        """Project (N, 1536) latents to (N, dim) PLS scores.

        Args:
            X: (n_features,) for a single star, or (N, n_features). Rows should
               already be star-aggregated (see ``aggregate_by_star``).
            dim: keep the first ``dim`` components; default all 16. Components
                 are nested, so this is equivalent to refitting with ``dim``.

        Returns:
            (dim,) for a single row, else (N, dim), float32.
        """
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        if X.shape[1] != self.x_mean.shape[0]:
            raise ValueError(f'Expected {self.x_mean.shape[0]} features, '
                             f'got {X.shape[1]}.')
        if dim is not None and not 1 <= dim <= self.n_components:
            raise ValueError(f'dim must be in 1..{self.n_components}, got {dim}.')

        rot = self.x_rotations if dim is None else self.x_rotations[:, :dim]
        T = ((X - self.x_mean) / self.x_std) @ rot
        T = T.astype(np.float32)
        return T[0] if single else T
