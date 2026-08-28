# EncoTESS data

This describes the data released with EncoTESS: the encoded (PCA) form of every light
curve, a 2-D UMAP map, and metadata tables for each light curve and each star. The metadata tables include both the specific fields included in training and supplementary stellar / light curve properties that are used elsewhere in the paper (e.g., $P_{rot}$, age).

## Files

**Included in the package** (`encotess/data/`):

| File | Rows | Contents |
|---|---|---|
| `latents_pca64.npz` | 69,293 | top-64 PCA encoding of every light curve (~98% of the variation) |
| `umap.npz` | 69,293 | 2-D UMAP map of every light curve |
| `encodings_pls3_star.npz` | 21,486 | **per-star** 3-component age view (matches the paper; see below) |
| `encodings_pls16_star.npz` | 21,486 | **per-star** 16-component age view (all labelled stars; see below) |
| `metadata_sector.csv` | 69,293 | metadata per light curve (one row per star–sector) |
| `metadata_FGKMcal_star.csv` | 11,238 | metadata per star, FGKM calibration sample |
| `metadata_hosts_star.csv` | 2,246 | metadata per star, exoplanet hosts |
| `metadata_thickdisk_star.csv` | 6,039 | metadata per star, thick-disk sample |

**Downloaded on demand** (hosted on HuggingFace, can be downloaded from there directly at https://huggingface.co/datasets/philvanlane/encotess/tree/main or via `encotess.assets`):

| File | Rows | Contents |
|---|---|---|
| `encodings_pca_full.npz` | 69,293 | the full 1536-component PCA encoding; array key: `latents_pca` |

## What a row means, and how to match rows up

- The **encodings, the UMAP, and `metadata_sector.csv` have one row per light curve** —
  one row per (star, sector) observation. A star observed in *N* sectors has *N* rows, each
  with its own encoding. These files are **lined up row-for-row**: row *i* is the same light
  curve in every one of them.
- **Match them by row position, not by an ID.** We removed exact duplicate light curves, but
  `(gaia_id, sector)` still isn't unique because a handful of stars carry two TIC ids. So
  join the per-sector table to the encodings by row order.
- The **`*_star.csv` files have one row per star**, keyed by `GaiaDR3_ID` (unique within each
  file). To attach star-level properties to a light curve, match
  `metadata_sector.csv.GaiaDR3_ID` to the right star table.

For convenience the `.npz` files also carry matching `gaia_ids`, `tic_ids`, `sectors`, and
`bank` (which sample the star belongs to) arrays.

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
| `iqr_half_flux` | instrumental | half the interquartile range of the flux |
| `camera` | 1–4 | TESS camera |
| `ccd` | 1–4 | TESS CCD |
| `cadence_s` | s | observing cadence (120 = 2-minute) |

### `metadata_FGKMcal_star.csv` (one row per star)

| Column | Units | Meaning |
|---|---|---|
| `GaiaDR3_ID` | — | Gaia DR3 source id |
| `age_Myr` | Myr | stellar age (see *Ages* below); blank for stars we only have a rotation period for |
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
from a single source), and `age_Myr` is the exoplanet-archive stellar age. Every host in this
file has an age.

### `metadata_thickdisk_star.csv` (one row per star)

`GaiaDR3_ID`, `BPRP0`, `BPRP0_err`, `G0`, `G0_err`, `parallax`, `parallax_error`, and
`dist_pc` (distance in parsecs). We don't release ages, rotation periods, or absolute
magnitudes for the thick-disk sample.

### `encodings_pls3_star.npz` and `encodings_pls16_star.npz` (one row per star)

Two low-dimensional "age views" of the latent, one row per star. Both files have the same
keys:

| Key | Shape | Meaning |
|---|---|---|
| `gaia_ids` | (21486,) | Gaia DR3 source id |
| `bank` | (21486,) | which sample the star belongs to (`FGKMcal` / `hosts` / `thickdisk`) |
| `pls` | (21486, K) | the K component scores (K = 3 or 16) |
| `in_age_fit` | (21486,) | whether this star was used to fit the view |

**How they were built.** For each star we combine its latents across all its sectors (taking
the element-wise maximum), then find a small set of directions that best track age — a PLS
(partial least squares) fit against `log10(age/Myr)`. We fit those directions once, on the
labelled stars, and then apply them to every star. These are **descriptive summaries of the
stars they were fit on** — good for seeing age-related structure in the latents — and **not**
an age predictor you should trust on new stars (careful, validated age prediction is a
separate model). The two files differ only in which stars they were fit on and how many
directions they keep:

- **`encodings_pls3_star.npz`** (3 components) — fit on the **2,893** stars used by the age
  analysis in the paper (labelled FGKMcal stars that have a *literature* rotation period),
  using that analysis's age convention. Reproduces the paper.
- **`encodings_pls16_star.npz`** (16 components) — fit on **all 9,221** labelled stars with an
  age (no rotation-period requirement), using the median age. Broader coverage, with extra
  directions in case the finer ones are useful. The leading direction is essentially the same
  in both files; the later ones differ.

## Where the values come from

- **Gaia parameters** (`BPRP0`, `G0`, `parallax`, and their errors) come from a Gaia DR3
  crossmatch on `GaiaDR3_ID`. Colours and magnitudes are **dereddened**; parallaxes are
  **zero-point corrected**. They agree between the light-curve files and the star tables to
  floating-point precision.
- **Ages (`metadata_FGKMcal_star.csv`).** A star's age is the **median of the ages given by
  the literature sources** for that star; `num_refs` counts those sources and `ref` lists
  them. The sources include: ChronoFlow, Feinstein 2024, Newton 2016/2018, Kiman 2021, MOCA,
  Mamonova 2025, Magaudda 2022, Pass 2022/2023/2024, the TIME table, and LWRD.
- **Ages (`metadata_hosts_star.csv`).** Host ages are the NASA Exoplanet Archive stellar age
  (`st_age`), converted from Gyr to Myr. These are the ages used in the host-star age
  analysis.
- **Absolute magnitude `MG`.** Distance-based: `MG = G0 − 5·log10(dist_pc/10)`, using a
  distance estimate rather than inverting the parallax. (This is the `MG` used in the
  rotation–colour–magnitude analysis; it is not the quick parallax-inversion version.)
- **Rotation periods.** `prot_lit` comes from the literature; `prot_tars` is measured directly
  from the TESS light curves. Both are per-star.
- **Flares and flux statistics** (`num_flares`, `ED_flare`, `skew_flux`, `kurt_flux`,
  `median_flux`, `iqr_half_flux`) are measured per sector from the light curves.

## The three samples

- **FGKMcal** — FGKM calibration stars (field and cluster stars with ages and/or rotation
  periods; the labelled stars used to calibrate age inference).
- **hosts** — TESS-observed exoplanet host stars.
- **thickdisk** — a kinematically selected thick-disk sample.

The encodings, UMAP, and per-light-curve metadata cover all three together; the per-star
tables are split by sample.

## What was left out

We started from 69,319 light curves. Two were dropped for a bad sector number (1751), and 24
were exact `(gaia_id, tic_id, sector)` duplicates (we kept the first of each pair) — leaving
**69,293** light curves in every released file.
