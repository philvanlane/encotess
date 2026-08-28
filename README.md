# EncoTESS

[![arXiv](https://img.shields.io/badge/arXiv-2608.25019-b31b1b.svg)](https://arxiv.org/abs/2608.25019)

**EncoTESS** turns a 2-minute-cadence TESS light curve into a fixed-length numerical
summary — a 1536-value "latent" vector that captures the shape of the star's brightness
variations. From that summary you can compare stars, explore structure across the sample,
or forecast the light curve a few steps ahead.

> ⚠️ **This repository is under active development** and its contents may still change.
> When the accompanying paper is published, a fixed, citable version will be archived on
> [Zenodo](https://zenodo.org/) with a DOI. Until then, please treat the code and data here
> as a preview.

## Install

```bash
pip install -e .
```

Needs `torch`, `numpy`, `h5py`, and `zuko`.

## Quick start

```python
import encotess

# 1. Encode a light curve -> 1536-value latent (+ its top-16 PCA)
enc = encotess.load_encoder()
z   = enc.encode(flux, flux_err, time, metadata=meta)   # arrays + a 13-field dict
z16 = enc.project_pca(z, dim=16)

# 2. Forecast flux a few steps ahead, with a 16th-84th percentile band
pred = encotess.predict_flux(enc, flux, flux_err, time, metadata=meta)
#   -> {'flux': median, 'p16': ..., 'p84': ..., 'time': ...}
```

`meta` is a dict of the 13 metadata fields (`encotess.DEFAULT_METADATA_FIELDS`). You can
leave out any you don't have (or set them to NaN) — the encoder was trained to cope with
missing fields. There are short runnable scripts in `examples/` and step-by-step notebooks
in `tutorials/`.

## The encodings

Alongside the model, we release the encoded form of every light curve in our sample. The
raw 1536-value latents are highly redundant, so we rotate them into a **PCA basis**: the
same information, re-ordered so the most informative directions come first. The full
version keeps all 1536 directions and loses nothing — it's just a tidier set of
coordinates.

Two versions ship for every light curve:

- **Comes with the package:** `encotess/data/latents_pca64.npz` — the top 64 PCA directions
  (~17 MB). These hold ~98% of the variation, which is plenty for almost any use.
- **Full version (download):** all 1536 PCA directions (~400 MB), from HuggingFace. This is
  an exact, reversible rewrite of the latent — grab it only if you need every last detail.

```python
import numpy as np
from encotess import assets
d = np.load(assets.pca_preview_path())
d['latents_pca64']                                     # (69293, 64) float32
d['gaia_ids'], d['tic_ids'], d['sectors'], d['bank']   # matching identifiers
```

**A UMAP map for plotting.** `encotess/data/umap.npz` holds a 2-D UMAP layout of every
light curve (`embedding`, shape `(69293, 2)`, with matching identifiers), so you can
reproduce the map of the sample directly.

**Per-star age views (PLS).** These are low-dimensional summaries of each star, built to
line up with stellar age, via `assets.pls_encoding_path(n_components=3|16)`: a 3-component
version (`encodings_pls3_star.npz`, matching the age analysis in the paper) and a
16-component version (`encodings_pls16_star.npz`, fit on all labelled stars). They're handy
for visualising age-related structure in the latents, but they are descriptive summaries of
the stars they were fit on — **not** a tested age predictor for new stars. See
[`DATASET.md`](DATASET.md).

**Metadata.** Ages, rotation periods, magnitudes, colours, parallaxes, and flare/flux
statistics ship as CSVs inside the package: one per-light-curve table
(`assets.metadata_path('sector')`, lined up row-for-row with the encodings) and three
per-star tables (`'FGKMcal_star'`, `'hosts_star'`, `'thickdisk_star'`). See
[`DATASET.md`](DATASET.md) for every column and where it comes from.

## What's in the box

| Path | Contents |
|---|---|
| `encotess/model.py` | the encoder network (a bidirectional MinGRU, ~95k parameters) |
| `encotess/encode.py` | `Encoder`: light curve → 1536-value latent (+ PCA) |
| `encotess/flux.py` | `predict_flux`: forecast flux a few steps ahead, with a p16–p84 band |
| `encotess/pca.py` | `GlobalPCA`: the PCA transform, in plain numpy |
| `encotess/weights/encotess_weights.pt` | the trained encoder |
| `encotess/weights/global_pca.npz` | the fitted PCA (all 1536 directions) |
| `encotess/data/latents_pca64.npz` | top-64 PCA encodings for every released light curve |
| `encotess/data/umap.npz` | 2-D UMAP layout for every released light curve |
| `encotess/data/encodings_pls{3,16}_star.npz` | per-star age views (PLS) |
| `encotess/data/metadata_sector.csv` | per-light-curve metadata (lined up with the encodings) |
| `encotess/data/metadata_{FGKMcal,hosts,thickdisk}_star.csv` | per-star metadata, by sample |

### The one thing you download

Everything above ships with the package. Only the full 1536-direction PCA encoding is too
big to bundle (~400 MB), so it's fetched on demand and cached locally:

```python
from encotess import assets
pca = assets.download_latents_pca()   # full 1536-direction PCA encoding (~400 MB)
```

Inside `encodings_pca_full.npz`, the array is stored under the key `latents_pca`. Most
people never need this — the bundled top-64 preview (`assets.pca_preview_path()`) already
covers ~98% of the variation. See [`DATASET.md`](DATASET.md) for details.

## License

MIT — see [LICENSE](LICENSE).
