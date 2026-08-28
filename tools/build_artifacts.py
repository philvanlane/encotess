"""Maintainer script: regenerate the shipped EncoTESS artifacts from the source
training outputs.

This is NOT needed to *use* EncoTESS — every artifact it produces is already
bundled in the package or downloadable. It exists to document provenance and to let
the maintainer rebuild the artifacts reproducibly. It reads the trained model
checkpoint and latent banks from the separate (unreleased) training repository, so
it only runs in that environment; the path constants below point into that repo.

Bundled in the wheel (small):

    encotess/weights/encotess_weights.pt   (trained encoder checkpoint)
    encotess/weights/global_pca.npz        (full-rank UNWHITENED global PCA, refit)
    encotess/data/latents_pca64.npz        (top-64 PCA encoding, all light curves)
    encotess/data/umap.npz                 (2-D UMAP embedding, all light curves)

External host (HuggingFace); built into dist/ (gitignored), uploaded after building:

    dist/encodings_pca_full.npz            (full 1536-d PCA encoding, ~400 MB)
    dist/metadata_per_sector.csv           (per-light-curve metadata, ~14 MB)
    dist/metadata_per_star.csv             (per-star metadata, ~4.5 MB)

Usage:
    python tools/build_artifacts.py --steps encoder,pca,pca_preview,umap
    python tools/build_artifacts.py --steps metadata,pca_full   # external (HF) artifacts
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- paths (into the separate training repository) -------------------------
REPO = Path(__file__).resolve().parents[2]           # training repo root
PKG = Path(__file__).resolve().parents[1] / 'encotess'  # the importable package dir
WEIGHTS = PKG / 'weights'
DATA = PKG / 'data'

# Trained encoder checkpoint + the latent banks it produced. The plain (non-merged)
# banks are the encoder-native banks that encode.py reproduces bit-for-bit.
MODEL_SRC = REPO / 'final_model/sendit/e100/model.pt'
# Reference PCA cache: the canonical standardization (X_mean/X_std) + the deployed
# top-16 basis that the full-rank refit is checked against.
PCA_D16_SRC = REPO / 'final_model/sendit/e100/age_inference/shared/global_pca_d16.npz'
LATENT_BASE = REPO / 'final_model/sendit/e100'
UMAP_SRC = LATENT_BASE / 'umap/umap_age_best_model_multiscale_data.npz'
DIST = PKG.parent / 'dist'          # HF-bound artifacts (outside the package; gitignored)
PREVIEW_DIM = 64                    # bundled compact PCA preview (~97.9% variance)

# Shipped encodings = the PLAIN (encoder-native) banks. The full-rank PCA is REFIT on
# the MERGED banks (s97/98 split) that the deployed d16 was fit on -> identical basis.
BANKS = [
    ('pretrain', REPO / 'final_pretrain/timeseries_pretrain.h5',
     LATENT_BASE / 'latents_pretrain.npz'),
    ('hosts', REPO / 'final_pretrain/timeseries_exop_hosts.h5',
     LATENT_BASE / 'latents_hosts.npz'),
    ('thickdisk', REPO / 'final_pretrain/timeseries_thickdisk.h5',
     LATENT_BASE / 'latents_thickdisk.npz'),
]
MERGED_BANKS = [LATENT_BASE / f'latents_{b}_merged.npz'
                for b in ('pretrain', 'hosts', 'thickdisk')]

# H5 metadata fields to carry into the metadata CSVs (beyond npz identifiers).
H5_META_EXTRA = ['Tmag', 'cadence_s', 'camera', 'ccd', 'parallax', 'parallax_error',
                 'G0', 'G0_err', 'median_flux', 'iqr_half_flux']

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import encotess
from encotess.pca import GlobalPCA  # noqa: E402


# --- steps -----------------------------------------------------------------
def step_encoder():
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    dst = WEIGHTS / 'encotess_weights.pt'
    shutil.copyfile(MODEL_SRC, dst)
    print(f'[encoder] {MODEL_SRC} -> {dst} ({dst.stat().st_size/1e6:.1f} MB)')


def step_pca(chunk=4000):
    """Refit the global PCA to FULL rank (1536 comps), UNWHITENED, and verify that
    its top-16 reproduces the deployed d16 basis exactly.

    Memory-safe: standardizes with the reference cache's X_mean/X_std (the canonical
    standardization) and accumulates the 1536x1536 covariance in chunks, so it never
    holds the whole 70,438x1536 matrix. Eigendecomposition of the covariance gives
    the PCA components (== sklearn PCA up to sign; svd_flip applied)."""
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    d16 = np.load(PCA_D16_SRC, allow_pickle=True)
    X_mean = d16['X_mean'].astype(np.float64)
    X_std = d16['X_std'].astype(np.float64)
    P = X_mean.size

    S1 = np.zeros(P); S2 = np.zeros((P, P)); n = 0
    for p in MERGED_BANKS:
        X = np.load(p, allow_pickle=True)['latent_vectors']       # (nb, 1536) f32
        for s in range(0, len(X), chunk):
            xs = (X[s:s + chunk].astype(np.float64) - X_mean) / (X_std + 1e-8)
            S1 += xs.sum(0); S2 += xs.T @ xs; n += len(xs)
        del X
    pca_mean = S1 / n
    cov = (S2 - n * np.outer(pca_mean, pca_mean)) / (n - 1)
    evals, evecs = np.linalg.eigh(cov)                             # ascending
    evals = evals[::-1]; comps = evecs[:, ::-1].T                  # desc; rows=comps
    sign = np.sign(comps[np.arange(P), np.abs(comps).argmax(1)])   # svd_flip
    sign[sign == 0] = 1; comps *= sign[:, None]
    evr = evals / evals.sum()

    # verify top-16 reproduces the deployed d16 basis
    dots = np.abs((comps[:16] * d16['pca_components']).sum(1))
    if dots.min() < 0.9999 or not np.allclose(evr[:16], d16['pca_explained_variance_ratio'], atol=1e-5):
        raise RuntimeError(f'full-rank refit does not match d16 (min|dot|={dots.min():.5f})')

    dst = WEIGHTS / 'global_pca.npz'
    np.savez_compressed(
        dst,
        X_mean=X_mean.astype(np.float32), X_std=X_std.astype(np.float32),
        pca_mean=pca_mean.astype(np.float32),
        pca_components=comps.astype(np.float32),
        pca_explained_variance=evals.astype(np.float32),
        pca_explained_variance_ratio=evr.astype(np.float32),
        pca_whiten=np.bool_(False),
        pca_n_components=np.int64(P), pca_n_features_in=np.int64(P),
        pca_n_samples=np.int64(n),
    )
    cum = np.cumsum(evr)
    print(f'[pca] refit full-rank UNWHITENED PCA on {n} merged rows -> {dst} '
          f'({dst.stat().st_size/1e6:.1f} MB); top-16 == d16 (min|dot|={dots.min():.4f}); '
          f'cum var @16={cum[15]:.3f} @64={cum[63]:.3f} @1536={cum[-1]:.3f}')


def _bank_frames():
    """Yield (bank_name, per_sector DataFrame) for each latent bank."""
    import h5py
    for name, h5_path, npz_path in BANKS:
        z = np.load(npz_path, allow_pickle=True)
        n = z['latent_vectors'].shape[0]
        cols = {
            'bank': name,
            'gaia_id': z['gaia_ids'].astype(str),
            'tic_id': z['tic_ids'].astype(np.int64),
            'sector': z['sectors'].astype(np.int64),
            'age_myr': z['ages'].astype(np.float64),
            'bprp0': z['bprp0'].astype(np.float64),
            'bprp0_err': z['bprp0_err'].astype(np.float64),
            'mg': z['mg'].astype(np.float64),
            'mg_err': z['mg_err'].astype(np.float64),
            'mem_prob': z['mem_prob'].astype(np.float64),
        }
        with h5py.File(h5_path, 'r') as f:
            meta = f['metadata']
            # sanity: npz row order matches H5 row order
            if not np.array_equal(meta['tic'][:].astype(np.int64), cols['tic_id']):
                raise RuntimeError(f'{name}: npz/H5 row order mismatch')
            for fld in H5_META_EXTRA:
                cols[fld] = (meta[fld][:].astype(np.float64) if fld in meta
                             else np.full(n, np.nan))
        yield name, pd.DataFrame(cols)


def step_metadata():
    """Per-light-curve + per-star metadata CSVs -> dist/ (HuggingFace; not bundled)."""
    DIST.mkdir(parents=True, exist_ok=True)
    frames = [df for _, df in _bank_frames()]
    per_sector = pd.concat(frames, ignore_index=True)
    ps_path = DIST / 'metadata_per_sector.csv'
    per_sector.to_csv(ps_path, index=False)
    print(f'[metadata] wrote {ps_path} — {len(per_sector)} light curves [upload to HF]')

    # Per-star: one row per gaia_id (latent_max-style: first static values + sector list).
    static = ['bank', 'tic_id', 'age_myr', 'bprp0', 'bprp0_err', 'mg', 'mg_err',
              'mem_prob'] + H5_META_EXTRA
    grp = per_sector.groupby('gaia_id', sort=False)
    per_star = grp.first()[static].copy()
    per_star['n_sectors'] = grp['sector'].count()
    per_star['sectors'] = grp['sector'].apply(
        lambda s: ' '.join(str(int(x)) for x in sorted(set(s))))
    per_star = per_star.reset_index()
    pstar_path = DIST / 'metadata_per_star.csv'
    per_star.to_csv(pstar_path, index=False)
    print(f'[metadata] wrote {pstar_path} — {len(per_star)} stars [upload to HF]')


def _project_plain_banks(dim):
    """Project the PLAIN latent banks through the bundled full-rank PCA to `dim`
    components (dim=None -> all 1536). Returns (Z, gaia, tic, sec, bank) concatenated
    in the [pretrain, hosts, thickdisk] order, memory-safe (one bank at a time)."""
    gp = GlobalPCA.from_npz(WEIGHTS / 'global_pca.npz')
    Z, gaia, tic, sec, bank = [], [], [], [], []
    for name, _, npz_path in BANKS:
        z = np.load(npz_path, allow_pickle=True)
        latv = z['latent_vectors']
        Z.append(gp.transform(latv, dim=dim).astype(np.float32))
        gaia.append(z['gaia_ids'].astype(str)); tic.append(z['tic_ids'].astype(np.int64))
        sec.append(z['sectors'].astype(np.int64)); bank.append(np.array([name] * len(latv)))
        print(f'  {name}: {latv.shape} -> {Z[-1].shape}')
        del latv, z
    return (np.concatenate(Z), np.concatenate(gaia), np.concatenate(tic),
            np.concatenate(sec), np.concatenate(bank))


def step_pca_preview():
    """Bundled compact encoding: top-PREVIEW_DIM PCA for every light curve."""
    DATA.mkdir(parents=True, exist_ok=True)
    Z, gaia, tic, sec, bank = _project_plain_banks(dim=PREVIEW_DIM)
    out = DATA / f'latents_pca{PREVIEW_DIM}.npz'
    np.savez_compressed(out, **{f'latents_pca{PREVIEW_DIM}': Z,
                                'gaia_ids': gaia, 'tic_ids': tic, 'sectors': sec, 'bank': bank})
    print(f'[pca_preview] wrote {out} — {Z.shape} ({out.stat().st_size/1e6:.2f} MB)')


def step_pca_full():
    """Full 1536-d PCA for every light curve -> dist/ (HuggingFace upload; not bundled)."""
    DIST.mkdir(parents=True, exist_ok=True)
    Z, gaia, tic, sec, bank = _project_plain_banks(dim=None)
    out = DIST / 'encodings_pca_full.npz'
    np.savez_compressed(out, encodings_pca=Z, gaia_ids=gaia, tic_ids=tic, sectors=sec, bank=bank)
    print(f'[pca_full] wrote {out} — {Z.shape} ({out.stat().st_size/1e6:.1f} MB) [upload to HF]')


def step_umap():
    """Bundle the 2-D UMAP embedding, aligned to the plain-bank concat order."""
    DATA.mkdir(parents=True, exist_ok=True)
    u = np.load(UMAP_SRC, allow_pickle=True)
    tic = np.concatenate([np.load(p, allow_pickle=True)['tic_ids'].astype(np.int64)
                          for _, _, p in BANKS])
    sec = np.concatenate([np.load(p, allow_pickle=True)['sectors'].astype(np.int64)
                          for _, _, p in BANKS])
    gaia = np.concatenate([np.load(p, allow_pickle=True)['gaia_ids'].astype(str)
                           for _, _, p in BANKS])
    bank = np.concatenate([np.array([name] * len(np.load(p, allow_pickle=True)['tic_ids']))
                           for name, _, p in BANKS])
    if not (np.array_equal(u['tic_ids'].astype(np.int64), tic)
            and np.array_equal(u['sectors'].astype(np.int64), sec)):
        raise RuntimeError('UMAP npz row order != plain-bank concat order')
    out = DATA / 'umap.npz'
    np.savez_compressed(out, embedding=u['embedding'].astype(np.float32),
                        gaia_ids=gaia, tic_ids=tic, sectors=sec, bank=bank)
    print(f'[umap] wrote {out} — {u["embedding"].shape} ({out.stat().st_size/1e6:.2f} MB)')


STEPS = {
    'encoder': step_encoder, 'pca': step_pca, 'metadata': step_metadata,
    'pca_preview': step_pca_preview, 'umap': step_umap, 'pca_full': step_pca_full,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps', default='encoder,pca,pca_preview,umap',
                    help='comma list from: ' + ','.join(STEPS))
    args = ap.parse_args()
    for name in args.steps.split(','):
        name = name.strip()
        if name not in STEPS:
            raise SystemExit(f'Unknown step {name!r}; choose from {list(STEPS)}')
        STEPS[name]()


if __name__ == '__main__':
    main()
