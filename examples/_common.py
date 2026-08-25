"""Shared helpers for the examples (a synthetic light curve + demo metadata)."""
import numpy as np


def synthetic_lightcurve(n=6000, period_days=3.2, cadence_min=2.0, seed=0):
    """A toy normalized TESS-like light curve (sinusoid + white noise).

    Not scientifically meaningful — just enough to exercise the API end to end.
    Flux is mean-subtracted / unit-scaled, matching the encoder's input regime.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * (cadence_min / 60.0 / 24.0)          # days
    flux = 0.8 * np.sin(2 * np.pi * t / period_days)
    flux = flux + rng.normal(0, 1.0, n)                     # dominated by noise
    flux = (flux - flux.mean()) / flux.std()
    flux_err = np.full(n, 1.0, dtype=np.float32)
    return flux.astype(np.float32), flux_err, t.astype(np.float32)


# A plausible set of the 13 metadata fields. Omit any you don't have (or set NaN);
# the encoder's mask channel handles missing fields.
DEMO_METADATA = {
    'cadence_s': 120.0, 'Tmag': 10.5, 'sector': 45, 'camera': 1, 'ccd': 2,
    'parallax': 5.0, 'parallax_error': 0.02, 'G0': 10.8, 'G0_err': 0.01,
    'BPRP0': 1.1, 'BPRP0_err': 0.03, 'median_flux': 1.0e4, 'iqr_half_flux': 50.0,
}
