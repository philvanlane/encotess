"""Encode a light curve into the 1536-d EncoTESS latent (+ top-16 PCA)."""
import encotess
from _common import synthetic_lightcurve, DEMO_METADATA

flux, flux_err, time = synthetic_lightcurve()

enc = encotess.load_encoder()                      # bundled EncoTESS encoder
z = enc.encode(flux, flux_err, time, metadata=DEMO_METADATA)
z16 = enc.project_pca(z, dim=16)

print(f"latent shape:      {z.shape}      (1536-d)")
print(f"pca-16 shape:      {z16.shape}       first 4 PCs: {z16[:4].round(3)}")
print(f"latent norm:       {(z ** 2).sum() ** 0.5:.2f}")
