# EncoTESS

[![arXiv](https://img.shields.io/badge/arXiv-2608.25019-b31b1b.svg)](https://arxiv.org/abs/2608.25019)

**EncoTESS** turns a 2-minute-cadence TESS light curve into a 1536-dim "latent" encoding. From that summary you can compare stars, explore structure across the sample, forecast the light curve a few steps ahead, or perform downstream tasks such as age inference.

> ⚠️ **This repository is still under development** and its contents may still change.
> When the accompanying paper is published, a fixed, citable version will be archived on
> [Zenodo](https://zenodo.org/) with a DOI.

## Install

```bash
pip install -e .
```

Needs `torch`, `numpy`, `h5py`, and `zuko`.

## Quick start

```python
import encotess

# 0. Prepare a light curve: download, robust-normalize, assemble the metadata
prep = encotess.preprocess_lightcurve(tic_id, sector, stellar_params=params)
flux, flux_err, time, meta = (prep['flux'], prep['flux_err'],
                              prep['time'], prep['metadata'])

# 1. Encode a light curve -> 1536-value latent (+ its top-16 PCA)
enc = encotess.load_encoder()
z   = enc.encode(flux, flux_err, time, metadata=meta)   # arrays + a 13-field dict
z16 = enc.project_pca(z, dim=16)
zp3 = enc.project_pls(z, dim=3)                         # age-covarying directions

# 3. Age posterior from those three components (demonstration model - see below)
post = encotess.load_age_model().infer(zp3, bprp0=1.1, bprp0_err=0.03)
#   -> {'median': ..., 'p16': ..., 'p84': ..., 'age_myr': ...} in log10(age/Myr)

# 2b. Forecast flux a few steps ahead, with a 16th-84th percentile band
pred = encotess.predict_flux(enc, flux, flux_err, time, metadata=meta)
#   -> {'flux': median, 'p16': ..., 'p84': ..., 'time': ...}
```

`meta` is a dict of the 13 metadata fields (`encotess.DEFAULT_METADATA_FIELDS`). You can
leave out any you don't have (or set them to NaN) — the encoder was trained to cope with
missing fields. Step 0 needs `pip install "encotess[preprocess]"`; the stellar parameters
it takes (`Tmag`, zero-point-corrected `parallax`, dereddened `G0`/`BPRP0`, and their
errors) are yours to supply — crossmatching and dereddening sit upstream of EncoTESS. There are short runnable scripts in `examples/` and step-by-step notebooks
in `tutorials/`.

## The encodings

Alongside the model, we release the encoded form of every light curve in our sample. The
raw 1536-value latents are highly redundant, so we rotate them into a **PCA basis**: the
same information, re-ordered so the most informative directions come first. The full
version keeps all 1536 directions.

We have provided two versions for every light curve:

- **Comes with the package:** `encotess/data/latents_pca64.npz` — the top 64 PCA directions
  (~17 MB). These hold ~98% of the variation, which should be sufficient for almost any use case.
- **Full version (download):** all 1536 PCA directions (~400 MB), from HuggingFace.

```python
import numpy as np
from encotess import assets
d = np.load(assets.pca_preview_path())
d['latents_pca64']                          # (69290, 64) float32
d['gaia_ids'], d['tic_ids'], d['sectors']   # identifiers: (TIC_ID, sector) is the key
```

**A UMAP map for plotting.** `encotess/data/umap.npz` holds a 2-D UMAP layout of every
light curve (`embedding`, shape `(69290, 2)`, with matching identifiers), so you can
reproduce the map of the sample directly.

**Per-star age views (PLS).** These are low-dimensional summaries of each star, built to
line up with stellar age, via `assets.pls_encoding_path()`: a 16-component view
(`encodings_pls16_star.npz`, fit on all labelled stars with an age). They're useful
for visualising age-related structure in the latents, but they are descriptive summaries of
the stars they were fit on — not a tested age predictor for new stars. See
[`DATASET.md`](DATASET.md).

**Metadata.** Ages, rotation periods, magnitudes, colours, parallaxes, and flare/flux
statistics are included in CSVs inside the package: one per-light-curve table
(`assets.metadata_path('sector')`, joined to the encodings on `(TIC_ID, sector)`) and three
per-star tables (`'FGKMcal_star'`, `'hosts_star'`, `'thickdisk_star'`, joined on
`GaiaDR3_ID`). See
[`DATASET.md`](DATASET.md) for every column and where it comes from.

## Components of EncoTESS

| Path | Contents |
|---|---|
| `encotess/preprocess.py` | raw TESS/Gaia inputs → encoder-ready arrays (download, robust flux normalization, metadata assembly) |
| `encotess/model.py` | the encoder network (a bidirectional MinGRU, ~95k parameters) |
| `encotess/encode.py` | `Encoder`: light curve → 1536-value latent (+ PCA) |
| `encotess/flux.py` | `predict_flux`: forecast flux a few steps ahead, with a p16–p84 band |
| `encotess/pca.py` | `GlobalPCA`: the PCA transform, in plain numpy |
| `encotess/pls.py` | `GlobalPLS`: the supervised PLS (age-covariance) projection, in plain numpy |
| `encotess/age.py` | `AgeNLE`: age posteriors from the PLS-3 scores (neural likelihood estimator) |
| `encotess/weights/encotess_weights.pt` | the trained encoder |
| `encotess/weights/global_pca.npz` | the fitted PCA (all 1536 directions) |
| `encotess/weights/pls_projection.npz` | the fitted PLS projection (16 age-covarying directions) |
| `encotess/weights/age_nle_pls3.pt` | the age likelihood model over the PLS-3 scores |
| `encotess/data/latents_pca64.npz` | top-64 PCA encodings for every released light curve |
| `encotess/data/umap.npz` | 2-D UMAP layout for every released light curve |
| `encotess/data/encodings_pls16_star.npz` | per-star age view (PLS, 16 components) |
| `encotess/data/metadata_sector.csv` | per-light-curve metadata (join to encodings on `(TIC_ID, sector)`) |
| `encotess/data/metadata_{FGKMcal,hosts,thickdisk}_star.csv` | per-star metadata, by sample |

### Downloading the full PCA encodings

Everything above is included in the package. Only the full 1536-direction PCA encoding is too
large (~400 MB), so it is fetched on demand and cached locally:

```python
from encotess import assets
pca = assets.download_latents_pca()   # full 1536-direction PCA encoding (~400 MB)
```

Inside `encodings_pca_full.npz`, the array is stored under the key `latents_pca`.

## License

MIT — see [LICENSE](LICENSE).
