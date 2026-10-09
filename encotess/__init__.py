"""EncoTESS: latent light-curve encoder for TESS 2-min photometry.

Encodes a variable-length TESS light curve into a fixed 1536-d latent
representation with a bidirectional MinGRU autoencoder, re-expresses that latent
in a global-PCA basis, and forecasts flux from the flow head.

Quick start
-----------
    import encotess
    enc  = encotess.load_encoder()
    z    = enc.encode(flux, flux_err, time, metadata=meta)   # 1536-d latent
    z16  = enc.project_pca(z, dim=16)                         # top-16 PCA
    rec  = encotess.predict_flux(enc, flux, flux_err, time, metadata=meta)

Raw TESS/Gaia inputs are prepared with ``encotess.preprocess`` (download, robust
flux normalization, metadata assembly) before ``encode`` sees them.

See the README for the model card and the shipped PCA-space encodings.
"""
from encotess.encode import Encoder, load_encoder
from encotess.flux import predict_flux
from encotess.pca import GlobalPCA
from encotess.pls import GlobalPLS, aggregate_by_star
from encotess.age import AgeNLE, load_age_model
from encotess.metadata import (
    MetadataStandardizer,
    DEFAULT_METADATA_FIELDS,
    ASTRO_METADATA_FIELDS,
    INSTRUMENTAL_METADATA_FIELDS,
)
from encotess import assets, preprocess
from encotess.preprocess import (
    download_lightcurve,
    clean_lightcurve,
    robust_normalize,
    build_metadata,
    preprocess_lightcurve,
    DEFAULT_QUALITY_BITMASK,
    STELLAR_PARAM_FIELDS,
)

__version__ = "0.1.0"

__all__ = [
    "Encoder",
    "load_encoder",
    "predict_flux",
    "GlobalPCA",
    "GlobalPLS",
    "AgeNLE",
    "load_age_model",
    "aggregate_by_star",
    "MetadataStandardizer",
    "DEFAULT_METADATA_FIELDS",
    "ASTRO_METADATA_FIELDS",
    "INSTRUMENTAL_METADATA_FIELDS",
    "assets",
    "preprocess",
    "download_lightcurve",
    "clean_lightcurve",
    "robust_normalize",
    "build_metadata",
    "preprocess_lightcurve",
    "DEFAULT_QUALITY_BITMASK",
    "STELLAR_PARAM_FIELDS",
    "__version__",
]
