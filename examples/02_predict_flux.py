"""Predict flux (offset-k forecast with a p16-p84 band) from a light curve."""
import encotess
from _common import synthetic_lightcurve, DEMO_METADATA

flux, flux_err, time = synthetic_lightcurve()

enc = encotess.load_encoder()
pred = encotess.predict_flux(enc, flux, flux_err, time, metadata=DEMO_METADATA,
                             offset=8, n_samples=64)

print(f"predicted points:  {len(pred['flux'])}")
print(f"median flux[:5]:   {pred['flux'][:5].round(3)}")
print(f"p16   flux[:5]:    {pred['p16'][:5].round(3)}")
print(f"p84   flux[:5]:    {pred['p84'][:5].round(3)}")
print(f"time  [:5]:        {pred['time'][:5].round(3)}")
print("\n(offset-k is the model's trained horizon; the band is its predictive "
      "uncertainty at k=8 steps.)")
