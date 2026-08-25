# EncoTESS

Encoder for 2-minute TESS light curves.

EncoTESS encodes a variable-length TESS light curve into a 1536-d latent
representation. This package includes the trained
encoder, the full-rank global PCA, and the PCA-space encodings + UMAP
of every released light curve (with per-light-curve metadata via on-demand download).

## Install

```bash
pip install -e .
```

Runtime dependencies: `torch`, `numpy`, `h5py`, `zuko`.

## Quick start

```python
import encotess

# 1. Encode a light curve -> 1536-d latent (+ top-16 PCA)
enc = encotess.load_encoder()
z   = enc.encode(flux, flux_err, time, metadata=meta)   # arrays + a 13-field dict
z16 = enc.project_pca(z, dim=16)

# 2. Predict flux (offset-k forecast with a p16-p84 band)
pred = encotess.predict_flux(enc, flux, flux_err, time, metadata=meta)
#   -> {'flux': median, 'p16': ..., 'p84': ..., 'time': ...}
```

`meta` is a dict of the 13 metadata fields (`encotess.DEFAULT_METADATA_FIELDS`);
missing fields may be omitted or set to NaN — the encoder was trained with a mask
channel and handles them. Runnable examples are in `examples/`.

## PCA-space encodings

The latents are provided in a **PCA basis**: a full-rank (1536-component), **un-normalized**
global PCA fit. Two forms are provided for every light curve:

- **Bundled, compact:** `encotess/data/latents_pca64.npz` — the top-64 PCA (~17 MB,
  ≈97.9% of the variance; effectively lossless for most uses).
- **External, complete:** the full 1536-d PCA (~400 MB) downloaded on demand from
  HuggingFace (see below).

```python
import numpy as np
from encotess import assets
d = np.load(assets.pca_preview_path())
d['latents_pca64']   # (69319, 64) float32
d['gaia_ids'], d['tic_ids'], d['sectors'], d['bank']   # aligned identifiers
```

**UMAP for plotting.** `encotess/data/umap.npz` holds the 2-D UMAP embedding for every
light curve (`embedding` (69319, 2) + aligned identifiers), so the released sample can be
replotted directly.

**Per-light-curve metadata** (magnitudes, colours, parallax, instrument fields, ...) is
hosted externally (HuggingFace): `assets.download_metadata('per_sector')` (one row per
light curve) and `assets.download_metadata('per_star')` (one row per star). See below.

## What's in the box

| Path | Contents |
|---|---|
| `encotess/model.py` | BiDirectional MinGRU encoder (95 k params) |
| `encotess/encode.py` | `Encoder`: light curve → 1536-d latent (+ PCA) |
| `encotess/flux.py` | `predict_flux`: offset-k flow forecast + p16/p84 band |
| `encotess/pca.py` | `GlobalPCA`: numpy full-rank global-PCA transform |
| `encotess/weights/encotess_weights.pt` | trained encoder |
| `encotess/weights/global_pca.npz` | full-rank (1536) un-normalized global PCA |
| `encotess/data/latents_pca64.npz` | top-64 PCA encodings for all released light curves |
| `encotess/data/umap.npz` | 2-D UMAP embedding for all released light curves |

### External downloads (HuggingFace)

Some artifacts need to be downloaded on demand into a local cache:

```python
from encotess import assets
pca       = assets.download_latents_pca()          # full 1536-d PCA encoding (~400 MB)
meta      = assets.download_metadata('per_sector') # per-light-curve metadata CSV (~14 MB)
meta_star = assets.download_metadata('per_star')   # per-star metadata CSV (~4.5 MB)
```

`encodings_pca_full.npz` is the PCA representation of the raw 1536-d latent. A subset of the top 64 PCA components can be accessed using `assets.pca_preview_path()`.

## License

MIT — see [LICENSE](LICENSE).
