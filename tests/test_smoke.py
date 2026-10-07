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
    # (TIC_ID, sector) is the per-light-curve primary key
    keys = list(zip(z['tic_ids'].tolist(), z['sectors'].tolist()))
    assert len(keys) == len(set(keys))


def test_umap_bundle():
    # The shipped 2-D UMAP embedding loads with numpy only, aligned to identifiers.
    z = np.load(assets.umap_path(), allow_pickle=False)
    assert z['embedding'].shape[1] == 2
    assert z['embedding'].shape[0] == z['tic_ids'].shape[0]


def test_pls_bundles():
    # Both per-star PLS encodings load with numpy only, keyed by a unique Gaia id.
    for k in (16,):
        z = np.load(assets.pls_encoding_path(), allow_pickle=False)
        assert z['pls'].shape[1] == k
        assert z['pls'].shape[0] == z['gaia_ids'].shape[0]
        assert len(np.unique(z['gaia_ids'])) == z['gaia_ids'].shape[0]  # GaiaDR3_ID is the key
        assert z['in_age_fit'].dtype == bool


def test_metadata_bundles():
    # Metadata CSVs are bundled (not downloaded) and load with the stdlib only.
    import csv
    n_lc = np.load(assets.pca_preview_path(), allow_pickle=False)['gaia_ids'].shape[0]
    with open(assets.metadata_path('sector'), newline='') as f:
        n_rows = sum(1 for _ in csv.reader(f)) - 1  # minus header
    assert n_rows == n_lc  # per-light-curve table has one row per encoding (same count)
    for which in ('FGKMcal_star', 'hosts_star', 'thickdisk_star'):
        p = assets.metadata_path(which)
        assert p.exists() and p.stat().st_size > 0


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


def test_preprocess_clean_and_normalize():
    # Offline: no network. Builds a raw-flux curve with the defects clean_lightcurve
    # and robust_normalize are there to absorb.
    from encotess import preprocess as pp
    rng = np.random.default_rng(0)
    n = 2000
    t = 2459000.0 + np.arange(n) * (2.0 / 60.0 / 24.0)
    f = 4256.0 + rng.normal(0, 11.0, n)
    fe = np.full(n, 10.0)
    f[5] = np.nan          # non-finite flux   -> cadence dropped
    t[7] = np.nan          # non-finite time   -> cadence dropped
    fe[11] = np.nan        # non-finite error  -> cadence kept, error filled
    f[100] = 1e6           # outlier           -> must not move the robust scale

    flux, flux_err, time = pp.clean_lightcurve(f, fe, t)
    assert len(flux) == n - 2
    assert time[0] == 0.0                       # time rebased to sector start
    assert np.isfinite(flux).all()
    assert not np.isfinite(flux_err).all()      # error NaN survives this step

    norm = pp.robust_normalize(flux, flux_err)
    assert np.isfinite(norm['flux_err']).all()  # ...and is filled by this one
    # Half the p16-p84 range equals sigma for Gaussian noise, and the 1e6 outlier
    # must not drag it (mean/std would move).
    assert abs(norm['median_flux'] - 4256.0) < 1.0
    assert abs(norm['iqr_half_flux'] - 11.0) < 1.0
    assert abs(np.median(norm['flux'])) < 1e-5
    assert norm['flux'].dtype == np.float32


def test_preprocess_metadata_roundtrip():
    from encotess import preprocess as pp
    meta = pp.build_metadata(sector=15, camera=2, ccd=4,
                             median_flux=4256.0, iqr_half_flux=11.0,
                             Tmag=11.4, parallax=41.35, parallax_error=0.018,
                             G0=12.42, G0_err=0.01, BPRP0=2.07, BPRP0_err=0.014)
    assert list(meta) == encotess.DEFAULT_METADATA_FIELDS   # order is load-bearing

    features, mask = pp.standardize(meta)
    assert features.shape == (13,) and mask.shape == (13,)
    assert mask.all()
    # standardize must agree with what encode() applies internally
    raw = {k: np.atleast_1d(np.float32(v)) for k, v in meta.items()}
    f2, m2 = encotess.MetadataStandardizer().transform(raw, return_mask=True)
    assert np.array_equal(f2[0], features) and np.array_equal(m2[0], mask)

    # Omitted fields are flagged missing; a log10 field cannot carry a negative
    # value, so a negative parallax is missing too, not silently wrong.
    sparse = pp.build_metadata(sector=15, camera=2, ccd=4, median_flux=4256.0,
                              iqr_half_flux=11.0, Tmag=11.4, BPRP0=-0.3,
                              parallax=-0.5)
    _, mask2 = pp.standardize(sparse)
    missing = {f for f, m in zip(encotess.DEFAULT_METADATA_FIELDS, mask2) if m == 0}
    assert 'G0' in missing and 'parallax' in missing
    assert 'BPRP0' not in missing          # raw field: negative colour is valid


def test_preprocess_rejects_bad_input():
    import pytest
    from encotess import preprocess as pp
    with pytest.raises(ValueError):
        pp.clean_lightcurve([1.0, 2.0], [1.0], [1.0, 2.0])          # ragged
    with pytest.raises(ValueError):
        pp.clean_lightcurve([np.nan] * 4, [1.0] * 4, [1.0] * 4)     # nothing finite
    with pytest.raises(ValueError):
        pp.robust_normalize(np.ones(100))                           # degenerate scale
    with pytest.raises(ValueError):
        pp.build_metadata(sector=1, camera=1, ccd=1, median_flux=1.0,
                          iqr_half_flux=1.0, Teff=5000.0)           # unknown field


def test_pls_projection_matches_released_view():
    # The bundled projection must reproduce the released pls16 scores. Done here
    # on the shipped artifacts only: check the affine form, nesting, and shapes.
    pls = encotess.GlobalPLS.from_npz(assets.pls_projection_path())
    assert pls.n_components == 16
    assert pls.x_mean.shape == (1536,) and pls.x_std.shape == (1536,)
    assert pls.x_rotations.shape == (1536, 16)
    assert pls.aggregation == 'max'

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(8, 1536))
    T = pls.transform(Z)
    assert T.shape == (8, 16) and T.dtype == np.float32

    # the documented affine form is exactly what transform does
    manual = ((Z - pls.x_mean) / pls.x_std) @ pls.x_rotations
    assert np.allclose(manual, T, atol=1e-4)

    # components are nested: dim=k truncates rather than refits
    for k in (1, 3, 8, 16):
        assert np.array_equal(pls.transform(Z, dim=k), T[:, :k])

    # unlike PCA this is NOT an orthonormal rotation
    G = pls.x_rotations.T @ pls.x_rotations
    assert not np.allclose(G, np.eye(16), atol=1e-6)

    # single-row input returns a 1-D vector
    assert pls.transform(Z[0], dim=4).shape == (4,)


def test_pls_rejects_bad_input():
    import pytest
    pls = encotess.GlobalPLS.from_npz(assets.pls_projection_path())
    with pytest.raises(ValueError):
        pls.transform(np.zeros((2, 64)))            # wrong feature count
    with pytest.raises(ValueError):
        pls.transform(np.zeros((2, 1536)), dim=17)  # beyond the fitted components
    with pytest.raises(ValueError):
        pls.transform(np.zeros((2, 1536)), dim=0)


def test_aggregate_by_star():
    rng = np.random.default_rng(1)
    gids = np.array(['a', 'a', 'a', 'b', 'c', 'c'])
    L = rng.normal(size=(6, 1536))
    uniq, Z = encotess.aggregate_by_star(L, gids, how='max')
    assert list(uniq) == ['a', 'b', 'c'] and Z.shape == (3, 1536)
    assert np.allclose(Z[0], L[:3].max(axis=0))     # released view uses max
    assert np.allclose(Z[1], L[3])                  # single-sector star passes through
    _, Zm = encotess.aggregate_by_star(L, gids, how='mean')
    assert np.allclose(Zm[2], L[4:6].mean(axis=0))
    import pytest
    with pytest.raises(ValueError):
        encotess.aggregate_by_star(L, gids, how='sum')
    with pytest.raises(ValueError):
        encotess.aggregate_by_star(L, gids[:3], how='max')


def test_encoder_project_pls():
    enc = encotess.load_encoder(device='cpu')
    flux, ferr, t = _synthetic()
    z = enc.encode(flux, ferr, t, metadata=META)
    assert enc.project_pls(z).shape == (16,)
    assert enc.project_pls(z, dim=3).shape == (3,)
    assert enc.project_pls(np.stack([z, z]), dim=5).shape == (2, 5)
