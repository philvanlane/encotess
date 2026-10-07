# EncoTESS data

This describes the data released with EncoTESS: the encoded (PCA) form of every light
curve, a 2-D UMAP map, and metadata tables for each light curve and each star. The metadata tables include both the specific fields included in training and supplementary stellar / light curve properties that are used elsewhere in the paper (e.g., $P_{rot}$, age).

## Files

**Included in the package** (`encotess/data/`):

| File | Rows | Contents |
|---|---|---|
| `latents_pca64.npz` | 69,290 | top-64 PCA encoding of every light curve (~98% of the variation) |
| `umap.npz` | 69,290 | 2-D UMAP map of every light curve |
| `encodings_pls16_star.npz` | 21,476 | **per-star** 16-component age view (all labelled stars; see below) |
| `metadata_sector.csv` | 69,290 | metadata per light curve (one row per star–sector) |
| `metadata_FGKMcal_star.csv` | 11,517 | metadata per star, FGKM calibration sample |
| `metadata_hosts_star.csv` | 3,948 | metadata per star, exoplanet hosts |
| `metadata_thickdisk_star.csv` | 6,020 | metadata per star, thick-disk sample |

**Downloaded on demand** (hosted on HuggingFace, can be downloaded from there directly at https://huggingface.co/datasets/philvanlane/encotess/tree/main or via `encotess.assets`):

| File | Rows | Contents |
|---|---|---|
| `encodings_pca_full.npz` | 69,290 | the full 1536-component PCA encoding; array key: `latents_pca` |

## What a row means, and how to join the files

- The **encodings, the UMAP, and `metadata_sector.csv` have one row per light curve** —
  one row per (star, sector) observation. A star observed in *N* sectors has *N* rows, each
  with its own encoding.
- **The primary key for a light curve is `(TIC_ID, sector)`.** These two fields together are
  unique across every per-light-curve file, so use them to join. (Each `GaiaDR3_ID` is also
  carried, but on its own it is *not* unique per light curve: a star observed in several
  sectors has several rows, and a few stars carry more than one TIC id.)
- The **`*_star.csv` files have one row per star**, keyed by `GaiaDR3_ID` (unique within each
  file). To attach star-level properties to a light curve, match
  `metadata_sector.csv.GaiaDR3_ID` to the right star table.

Always join on these keys rather than on row order. The `.npz` files carry `gaia_ids`,
`tic_ids`, and `sectors` arrays that label each encoding row, so you can join an encoding to
either table by its key — don't assume two files share the same row order. (They happen to be
row-aligned today, but the keys are what's guaranteed.) A star's **sample** (FGKMcal / hosts /
thick-disk) is not stored as a field anywhere — it is simply which star table the star appears
in.

## Columns

### `metadata_sector.csv` (one row per light curve)

| Column | Units | Meaning |
|---|---|---|
| `GaiaDR3_ID` | — | Gaia DR3 source id |
| `TIC_ID` | — | TESS Input Catalog id |
| `sector` | — | TESS observing sector |
| `Tmag` | mag | TESS magnitude |
| `skew_flux` | — | skewness of the sector's flux distribution |
| `kurt_flux` | — | kurtosis of the sector's flux distribution |
| `num_flares` | count | number of flares detected in the sector |
| `ED_flare` | — | total flare equivalent duration in the sector |
| `median_flux` | instrumental | median flux over the sector |
| `iqr_half_flux` | instrumental | the flux's 16–84 **semi-interpercentile range**, `(P84 − P16) / 2` — *not* half the interquartile range, despite the name (see note below) |
| `camera` | 1–4 | TESS camera |
| `ccd` | 1–4 | TESS CCD |
| `cadence_s` | s | observing cadence (120 = 2-minute) |

> **A note on `iqr_half_flux`.** The field name is a misnomer kept for compatibility with
> the released CSVs, HDF5 banks and encodings. The quantity is the **semi-interpercentile
> range** `(P84 − P16) / 2`, the direct analogue of the semi-interquartile range but
> taken at the 16th and 84th percentiles rather than the quartiles. It is a robust
> estimate of the flux's standard deviation: a Gaussian puts ±1σ at the 15.87th and
> 84.13th percentiles, so `P84 − P16 = 2σ` and the half-range equals σ, without a flare
> or a systematic inflating it. Computing half the actual interquartile range,
> `(P75 − P25) / 2`, gives roughly 0.67σ instead and will mis-scale every light curve.

### `metadata_FGKMcal_star.csv` (one row per star)

| Column | Units | Meaning |
|---|---|---|
| `GaiaDR3_ID` | — | Gaia DR3 source id |
| `age_Myr` | Myr | stellar age (see *Ages* below); blank when the star has no age (a rotation period only, or neither) |
| `num_refs` | count | how many literature sources gave an age for this star |
| `ref` | — | those sources, as a semicolon-separated list |
| `prot_lit` | days | rotation period from the literature |
| `prot_tars` | days | rotation period measured from the TESS light curves |
| `BPRP0` | mag | dereddened Gaia BP−RP colour |
| `BPRP0_err` | mag | uncertainty on `BPRP0` |
| `G0` | mag | dereddened Gaia G magnitude |
| `G0_err` | mag | uncertainty on `G0` |
| `parallax` | mas | Gaia DR3 parallax (zero-point corrected) |
| `parallax_error` | mas | uncertainty on `parallax` |
| `MG` | mag | absolute G magnitude (see *Absolute magnitude* below) |

### `metadata_hosts_star.csv` (one row per star)

Same columns as the FGKMcal star table but **without** `num_refs`/`ref` (each host age comes
from a single source), and `age_Myr` is the exoplanet-archive stellar age. Hosts without an
archive age are still included (with `age_Myr` blank) so that every encoded host has a row;
2,246 of the 3,948 have an age.

### `metadata_thickdisk_star.csv` (one row per star)

`GaiaDR3_ID`, `BPRP0`, `BPRP0_err`, `G0`, `G0_err`, `parallax`, `parallax_error`, and
`dist_pc` (distance in parsecs). We don't release ages, rotation periods, or absolute
magnitudes for the thick-disk sample.

### `encodings_pls16_star.npz` (one row per star)

Two low-dimensional "age views" of the latent, one row per star. Both files have the same
keys:

| Key | Shape | Meaning |
|---|---|---|
| `gaia_ids` | (21476,) | Gaia DR3 source id (the primary key) |
| `pls` | (21476, K) | the K component scores (K = 3 or 16) |
| `in_age_fit` | (21476,) | whether this star was used to fit the view |

**How they were built.** For each star we combine its latents across all its sectors (taking
the element-wise maximum), then find a small set of directions that best track age — a PLS
(partial least squares) fit against `log10(age/Myr)`. We fit those directions once, on the
labelled stars, and then apply them to every star. These are **descriptive summaries of the
stars they were fit on** — good for seeing age-related structure in the latents — and **not**
an age predictor you should trust on new stars (careful, validated age prediction is a
separate model). 

Fit on the **9,221** labelled stars carrying an age, using the median age. Components are
nested, so keeping the top `k` of the 16 is exactly a `k`-component fit.

### Projecting your own latents (`weights/pls_projection.npz`)

The package also ships the **projection itself** for the 16-component view, so a new
1536-d latent can be placed in the same space. The transform is affine:

```
X_std = (X - x_mean) / x_std          # per-dimension centre AND scale
T     = X_std @ x_rotations[:, :k]    # (N, k) component scores
```

`x_mean` / `x_std` (both `(1536,)`) and `x_rotations` (`(1536, 16)`) come from
`sklearn.cross_decomposition.PLSRegression(n_components=16, scale=True)` fit against
`log10(age/Myr)`, over per-star latents aggregated with an element-wise **maximum**
across that star's sectors.

Exactly what it was fit on, for the record:

| | |
|---|---|
| Stars | the 9,221 flagged `in_age_fit` in `encodings_pls16_star.npz` |
| Equivalently | the FGKMcal stars that carry an age (9,221 of 11,517) |
| Excluded entirely | hosts and thick-disk stars — both fits are FGKMcal-only |
| Age values | the `age_Myr` column of `metadata_FGKMcal_star.csv` — the median of the literature ages |
| Target | `log10(age_Myr)` |
| Hosts / thick-disk stars in the fit | none |

The same values are stored inside `pls_projection.npz` (`age_source`, `fit_population`,
`target`, `aggregation`, `n_fit_stars`). Use `encotess.GlobalPLS` /
`Encoder.project_pls`, or the formula above.

Three properties worth knowing:

- **Components are nested.** The first `k` columns equal a `k`-component fit, so `dim=k`
  truncates rather than refits. One matrix serves every `k <= 16`.
- **`x_rotations` is not orthonormal.** This is a projection, not a rotation: it does not
  invert back to the latent, and there is no per-component explained variance.
- **Scores are raw**, on the scale of the released encodings (component 1 has a standard
  deviation of ~22). The age pipeline z-scores them per fold; that is a property of a
  fold, not of the projection, so it is not applied here.

This projection reproduces the released 16-component scores to a median residual of
3.3e-6 of a component standard deviation. To keep fewer directions, pass `dim=k` —
components are nested, so the top `k` are exactly what a `k`-component fit would give.

> **This projection is supervised, so it can leak.** It chose its directions using the
> ages of the `in_age_fit` stars. Do not use these scores as input features for an age
> model that you then evaluate on any star in that fit set — the evaluation would be
> optimistic. A leak-free pipeline refits PLS inside each CV fold on that fold's training
> split alone. Note that per-fold refitting controls *target* leakage only: sector
> structure lives in the latents and survives any split, so a single light curve's score
> still depends on which sector observed it.

## Where the values come from

- **Gaia parameters** (`BPRP0`, `G0`, `parallax`, and their errors) come from a Gaia DR3
  crossmatch on `GaiaDR3_ID`. Colours and magnitudes are **dereddened**; parallaxes are
  **zero-point corrected**. They agree between the light-curve files and the star tables to
  floating-point precision.
- **Ages (`metadata_FGKMcal_star.csv`).** A star's age is the **median of the ages given by
  the literature sources** for that star; `num_refs` counts those sources and `ref` lists
  them.
- **Ages (`metadata_hosts_star.csv`).** Host ages are the NASA Exoplanet Archive stellar age
  (`st_age`), converted from Gyr to Myr. These are the ages used in the host-star age
  analysis.
- **Absolute magnitude `MG`.** Distance-based: `MG = G0 − 5·log10(dist_pc/10)`, using a
  distance estimate from Bailer-Jones 2021 where possible.
- **Rotation periods.** `prot_lit` comes from the literature source; `prot_tars` is from the TARS catalog. Both are per-star.
- **Flares and flux statistics** (`num_flares`, `ED_flare`, `skew_flux`, `kurt_flux`,
  `median_flux`, `iqr_half_flux`) are measured per sector from the light curves.

## The three samples

- **FGKMcal** — the FGKM calibration sample (field and cluster stars). Most carry an age
  and/or a rotation period and are the labelled stars used to calibrate age inference; a small
  number carry neither and are included for completeness.
- **hosts** — TESS-observed exoplanet host stars.
- **thickdisk** — the kinematically selected thick-disk sample.

The encodings, UMAP, and per-light-curve metadata cover all three together; the per-star
tables are split by sample.
