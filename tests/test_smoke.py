"""Smoke tests: load the bundled artifacts and exercise each public entry point.

Fast by design — runs in seconds. Run with:  pytest -q
"""
import numpy as np

import encotess
from encotess import assets


def _synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * (2.0 / 60.0 / 24.0)
    flux = np.sin(2 * np.pi * t / 3.0) + rng.normal(0, 1.0, n)
    flux = (flux - flux.mean()) / flux.std()
    return flux.astype(np.float32), np.ones(n, np.float32), t.astype(np.float32)


META = {'cadence_s': 120.0, 'Tmag': 10.5, 'sector': 45, 'camera': 1, 'ccd': 2,
        'parallax': 5.0, 'parallax_error': 0.02, 'G0': 10.8, 'G0_err': 0.01,
        'BPRP0': 1.1, 'BPRP0_err': 0.03, 'median_flux': 1e4, 'iqr_half_flux': 50.0}


def test_version():
    assert isinstance(encotess.__version__, str)


def test_encode_and_pca():
    enc = encotess.load_encoder(device='cpu')
    flux, ferr, t = _synthetic()
    z = enc.encode(flux, ferr, t, metadata=META)
    assert z.shape == (1536,)
    assert np.isfinite(z).all()
    z16 = enc.project_pca(z, dim=16)
    assert z16.shape == (16,)
    assert np.isfinite(z16).all()


def test_predict_flux():
    enc = encotess.load_encoder(device='cpu')
    flux, ferr, t = _synthetic()
    pred = encotess.predict_flux(enc, flux, ferr, t, metadata=META,
                                 offset=8, n_samples=16)
    assert len(pred['flux']) == len(pred['time'])
    assert np.isfinite(pred['flux']).all()
    assert (pred['p16'] <= pred['p84'] + 1e-6).all()


def test_pca_preview_bundle():
    # The shipped per-light-curve top-64 PCA preview loads with numpy only.
    z = np.load(assets.pca_preview_path(), allow_pickle=False)
    assert z['latents_pca64'].shape[1] == 64
    assert z['latents_pca64'].shape[0] == z['gaia_ids'].shape[0]


def test_umap_bundle():
    # The shipped 2-D UMAP embedding loads with numpy only, aligned to identifiers.
    z = np.load(assets.umap_path(), allow_pickle=False)
    assert z['embedding'].shape[1] == 2
    assert z['embedding'].shape[0] == z['tic_ids'].shape[0]


def test_full_pca_is_lossless_rotation():
    # Full-rank unwhitened PCA is an orthonormal rotation -> invertible to the latent.
    enc = encotess.load_encoder(device='cpu')
    flux, ferr, t = _synthetic()
    z = enc.encode(flux, ferr, t, metadata=META)          # 1536-d latent
    zf = enc.project_pca(z)                                # dim=None -> full 1536-d PCA
    assert zf.shape == (1536,)
    # reconstruct the standardized latent from the full PCA and invert standardization
    gp = enc._pca
    x_norm = zf.astype(np.float64) @ gp.components + gp.pca_mean
    x_rec = x_norm * (gp.X_std + 1e-8) + gp.X_mean
    assert np.allclose(x_rec, z, atol=1e-3)
