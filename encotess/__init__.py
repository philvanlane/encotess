"""EncoTESS: latent light-curve encoder for TESS 2-min photometry.

Encodes a variable-length TESS light curve into a fixed 1536-d latent
representation with a bidirectional MinGRU autoencoder, exposes the top-16
global-PCA projection of that latent, and forecasts flux from the flow head.

Quick start
-----------
    import encotess
    enc  = encotess.load_encoder()
    z    = enc.encode(flux, flux_err, time, metadata=meta)   # 1536-d latent
    z16  = enc.project_pca(z)                                 # top-16 PCA
    rec  = encotess.predict_flux(enc, flux, flux_err, time, metadata=meta)

See the README for the model card and the shipped PCA-space encodings.
"""
from encotess.encode import Encoder, load_encoder
from encotess.flux import predict_flux
from encotess.pca import GlobalPCA
from encotess.metadata import (
    MetadataStandardizer,
    DEFAULT_METADATA_FIELDS,
    ASTRO_METADATA_FIELDS,
    INSTRUMENTAL_METADATA_FIELDS,
)
from encotess import assets

__version__ = "0.1.0"

__all__ = [
    "Encoder",
    "load_encoder",
    "predict_flux",
    "GlobalPCA",
    "MetadataStandardizer",
    "DEFAULT_METADATA_FIELDS",
    "ASTRO_METADATA_FIELDS",
    "INSTRUMENTAL_METADATA_FIELDS",
    "assets",
    "__version__",
]
