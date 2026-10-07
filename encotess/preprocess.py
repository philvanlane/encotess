"""Raw TESS/Gaia inputs -> encoder-ready light curve and metadata.

``Encoder.encode`` expects a light curve that has already been through the same
preparation the training set went through. This module is that preparation, in
the order the released banks were built:

1. ``download_lightcurve`` — fetch a SPOC 2-min light curve from MAST and pull
   PDCSAP flux plus the instrumental header fields.
2. ``clean_lightcurve`` — drop non-finite samples and reset the time axis to the
   start of the sector.
3. ``robust_normalize`` — centre on the median and scale by the 16-84
   semi-interpercentile range, recording both as the ``median_flux`` /
   ``iqr_half_flux`` metadata fields the encoder is conditioned on.
4. ``build_metadata`` — assemble the 13 raw metadata fields from the header, the
   flux statistics, and the stellar parameters you supply.
5. ``standardize`` — apply the fixed log10/scale rules that turn those raw
   fields into the values the network actually sees (also applied internally by
   ``encode``; exposed here so you can inspect them).

``preprocess_lightcurve`` runs 1-4 in one call.

What this module does **not** do: the stellar parameters are taken as given.
Crossmatching a TIC to Gaia DR3, correcting the parallax zero point, and
dereddening the photometry to ``G0`` / ``BPRP0`` are upstream of EncoTESS and
are left to the caller — see ``build_metadata`` for the exact conventions the
released values follow.
"""
from __future__ import annotations

import numpy as np

from encotess.metadata import DEFAULT_METADATA_FIELDS, MetadataStandardizer

# TESS QUALITY bitmask applied at download time. 17087 is lightkurve's "default"
# mask as of lk 2.5.1; passing the integer explicitly pins the behaviour across
# lightkurve versions (the same keyword resolved to 175 in lk <= 2.5.0).
DEFAULT_QUALITY_BITMASK = 17087

# 2-min cadence, in seconds. Constant across the released banks.
CADENCE_S = 120.0

# Stellar fields the caller must supply; see build_metadata for conventions.
STELLAR_PARAM_FIELDS = [
    'Tmag', 'parallax', 'parallax_error', 'G0', 'G0_err', 'BPRP0', 'BPRP0_err',
]


def download_lightcurve(tic_id, sector, quality_bitmask=DEFAULT_QUALITY_BITMASK,
                        author='SPOC', exptime=120, flux_column='pdcsap_flux'):
    """Download one SPOC 2-min light curve from MAST.

    Requires ``lightkurve`` (``pip install 'encotess[preprocess]'``).

    Args:
        tic_id: TIC identifier (int or str, without the 'TIC ' prefix).
        sector: TESS sector number.
        quality_bitmask: TESS QUALITY bits to reject. Pass an integer rather
            than ``'default'`` so the result does not depend on the lightkurve
            version (see ``DEFAULT_QUALITY_BITMASK``).
        author, exptime, flux_column: pipeline, cadence, and flux column. The
            released banks used SPOC / 120 s / PDCSAP throughout.

    Returns:
        dict with raw ``flux``, ``flux_err``, ``time`` (BTJD days, untouched)
        and the header fields ``sector``, ``camera``, ``ccd``, ``cadence_s``.

    Raises:
        LookupError: no matching light curve for that TIC and sector.
    """
    import lightkurve as lk

    tic_id = int(tic_id)
    res = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS', sector=int(sector),
                               author=author, exptime=exptime)
    # MAST cone-searches: keep only the requested target, not its neighbours.
    res = res[res.table['target_name'] == str(tic_id)]
    if len(res) == 0:
        raise LookupError(f'No {author} {exptime}s light curve for TIC {tic_id} '
                          f'sector {sector}.')

    lc = res[0].download(quality_bitmask=quality_bitmask)
    return {
        'flux': np.asarray(lc[flux_column].value.filled(np.nan), dtype=np.float64),
        'flux_err': np.asarray(lc[f'{flux_column}_err'].value.filled(np.nan),
                               dtype=np.float64),
        'time': np.asarray(lc['time'].value, dtype=np.float64),
        'sector': int(sector),
        'camera': int(lc.meta['CAMERA']),
        'ccd': int(lc.meta['CCD']),
        'cadence_s': float(CADENCE_S),
    }


def clean_lightcurve(flux, flux_err, time, reset_time=True):
    """Drop non-finite samples and rebase the time axis.

    Keeps only cadences with finite flux *and* finite time, matching training.
    Non-finite ``flux_err`` values are kept here and filled in
    ``robust_normalize``, so an isolated bad error bar does not cost a cadence.

    Args:
        flux, flux_err, time: equal-length raw arrays.
        reset_time: subtract ``time[0]`` so the axis starts at 0 days, as the
            released H5 banks store it.

    Returns:
        (flux, flux_err, time) as float64 arrays.
    """
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if not (len(flux) == len(flux_err) == len(time)):
        raise ValueError('flux, flux_err and time must be the same length.')

    keep = np.isfinite(flux) & np.isfinite(time)
    if not keep.any():
        raise ValueError('No finite flux/time samples in this light curve.')
    flux, flux_err, time = flux[keep], flux_err[keep], time[keep]

    if reset_time:
        time = time - time[0]
    return flux, flux_err, time


def robust_normalize(flux, flux_err=None, fill_err_nan=True):
    """Centre on the median and scale by the 16-84 semi-interpercentile range.

    This is the normalization the released encoder was trained on::

        median_flux   = median(flux)
        iqr_half_flux = (p84 - p16) / 2
        flux          = (flux - median_flux) / iqr_half_flux
        flux_err      = flux_err / iqr_half_flux

    ``iqr_half_flux`` is a misnomer kept for compatibility with the released
    data: the quantity is the **semi-interpercentile range**, taken at the 16th
    and 84th percentiles, not half the interquartile range. It is a robust
    estimate of the flux's standard deviation — a Gaussian puts +/-1 sigma at
    the 15.87th and 84.13th percentiles, so the half-range equals sigma without
    a flare inflating it. Half the true IQR would give ~0.67 sigma instead.

    Both statistics are computed over finite flux values only, and both are fed
    back to the encoder as metadata, so the network knows the absolute scale it
    was normalized away from.

    Args:
        flux: raw flux (e.g. PDCSAP, in e-/s).
        flux_err: matching uncertainties, or None.
        fill_err_nan: replace non-finite normalized ``flux_err`` with that
            curve's median, as the released banks do. The encoder has no
            missing-value channel for ``flux_err``.

    Returns:
        dict with ``flux``, ``flux_err`` (float32, or None), and the float
        ``median_flux`` / ``iqr_half_flux``.

    Raises:
        ValueError: the p16-p84 range is zero or non-finite (degenerate curve).
    """
    flux = np.asarray(flux, dtype=np.float64)
    finite = np.isfinite(flux)
    if not finite.any():
        raise ValueError('No finite flux samples to normalize.')

    median_flux = float(np.median(flux[finite]))
    p84, p16 = np.percentile(flux[finite], [84, 16])
    iqr_half_flux = float((p84 - p16) / 2.0)
    if not np.isfinite(iqr_half_flux) or iqr_half_flux <= 0:
        raise ValueError(f'Degenerate iqr_half_flux={iqr_half_flux}; the flux is '
                         'constant or almost entirely non-finite.')

    out = {
        'flux': ((flux - median_flux) / iqr_half_flux).astype(np.float32),
        'flux_err': None,
        'median_flux': median_flux,
        'iqr_half_flux': iqr_half_flux,
    }

    if flux_err is not None:
        err = np.asarray(flux_err, dtype=np.float64) / iqr_half_flux
        if fill_err_nan:
            bad = ~np.isfinite(err)
            if bad.any():
                med = np.nanmedian(err)
                err[bad] = med if np.isfinite(med) else 0.0
        out['flux_err'] = err.astype(np.float32)
    return out


def build_metadata(sector, camera, ccd, median_flux, iqr_half_flux,
                   cadence_s=CADENCE_S, **stellar_params):
    """Assemble the 13 raw metadata fields the encoder is conditioned on.

    Instrumental fields come from the light-curve header, the two flux fields
    from ``robust_normalize``, and the rest are stellar parameters you supply.
    Values are stored raw here; ``standardize`` applies the log10/scale rules.

    Conventions the released metadata follows (match them, or the encoder sees
    a star it was never trained on):

    - ``Tmag``: TESS magnitude from the TIC.
    - ``parallax``: Gaia DR3 parallax in mas, **zero-point corrected**
      (Lindegren et al. 2021); ``parallax_error`` is the catalogue error, mas.
    - ``G0``, ``BPRP0``: apparent Gaia G magnitude and BP-RP colour,
      **dereddened**. ``G0_err`` / ``BPRP0_err`` are their propagated errors.
    - Fields you do not have may be omitted or passed as NaN; ``standardize``
      flags them in the mask channel and the encoder handles them.

    Note that ``parallax``, ``G0`` and the error fields are log10-transformed
    downstream, so non-positive values cannot be represented and are treated as
    missing. ``BPRP0`` is passed raw and may be negative.

    Args:
        sector, camera, ccd: instrumental fields from the light-curve header.
        median_flux, iqr_half_flux: from ``robust_normalize``.
        cadence_s: exposure time in seconds (120 for the released banks).
        **stellar_params: any of ``STELLAR_PARAM_FIELDS``.

    Returns:
        dict keyed by ``DEFAULT_METADATA_FIELDS``, ready for ``Encoder.encode``.

    Raises:
        ValueError: an unrecognized stellar parameter was passed.
    """
    unknown = set(stellar_params) - set(STELLAR_PARAM_FIELDS)
    if unknown:
        raise ValueError(f'Unknown stellar parameter(s): {sorted(unknown)}. '
                         f'Expected any of {STELLAR_PARAM_FIELDS}.')

    meta = {
        'cadence_s': float(cadence_s),
        'sector': float(sector),
        'camera': float(camera),
        'ccd': float(ccd),
        'median_flux': float(median_flux),
        'iqr_half_flux': float(iqr_half_flux),
    }
    for field in STELLAR_PARAM_FIELDS:
        value = stellar_params.get(field, np.nan)
        meta[field] = float(value) if value is not None else np.nan
    return {field: meta[field] for field in DEFAULT_METADATA_FIELDS}


def standardize(metadata, fields=None):
    """Standardize one metadata dict into the values the network sees.

    Thin single-curve wrapper over ``MetadataStandardizer`` (which ``encode``
    applies internally). Useful for inspecting what a field becomes, and for
    checking which fields were dropped as missing.

    Args:
        metadata: scalar-valued dict, e.g. from ``build_metadata``.
        fields: field order; defaults to ``DEFAULT_METADATA_FIELDS``.

    Returns:
        (features, mask) 1-D float32 arrays. ``features`` is NaN-free;
        ``mask`` is 0 where a field was missing or outside its transform's
        domain (non-positive under log10).
    """
    fields = fields or DEFAULT_METADATA_FIELDS
    raw = {f: np.atleast_1d(np.asarray(metadata.get(f, np.nan), dtype=np.float32))
           for f in fields}
    features, mask = MetadataStandardizer(fields=fields).transform(raw, return_mask=True)
    return features[0], mask[0]


def preprocess_lightcurve(tic_id, sector, stellar_params=None,
                          quality_bitmask=DEFAULT_QUALITY_BITMASK, **download_kwargs):
    """Download and fully prepare one light curve for ``Encoder.encode``.

    Runs ``download_lightcurve`` -> ``clean_lightcurve`` -> ``robust_normalize``
    -> ``build_metadata``.

    Args:
        tic_id, sector: the target.
        stellar_params: dict of any of ``STELLAR_PARAM_FIELDS``; omitted fields
            are treated as missing.
        quality_bitmask: see ``download_lightcurve``.
        **download_kwargs: forwarded to ``download_lightcurve``.

    Returns:
        dict with ``flux``, ``flux_err``, ``time`` (float32, encoder-ready) and
        ``metadata`` (the 13 raw fields).
    """
    raw = download_lightcurve(tic_id, sector, quality_bitmask=quality_bitmask,
                              **download_kwargs)
    flux, flux_err, time = clean_lightcurve(raw['flux'], raw['flux_err'], raw['time'])
    norm = robust_normalize(flux, flux_err)
    metadata = build_metadata(
        sector=raw['sector'], camera=raw['camera'], ccd=raw['ccd'],
        median_flux=norm['median_flux'], iqr_half_flux=norm['iqr_half_flux'],
        cadence_s=raw['cadence_s'], **(stellar_params or {}),
    )
    return {
        'flux': norm['flux'],
        'flux_err': norm['flux_err'],
        'time': time.astype(np.float32),
        'metadata': metadata,
    }
