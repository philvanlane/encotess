# Tutorials

Step-by-step notebooks that walk through what EncoTESS does. Each one is short and runs
top-to-bottom. Notebooks 2–5 use only the bundled data; notebook 1 downloads a light
curve from MAST and needs network access.

| Notebook | What it covers |
|---|---|
| [`01_preprocess_a_lightcurve.ipynb`](01_preprocess_a_lightcurve.ipynb) | Download a TESS light curve under an explicit quality bitmask, normalize the flux on the median and half the p16–p84 range, and assemble the metadata the encoder is conditioned on. |
| [`02_encode_a_lightcurve.ipynb`](02_encode_a_lightcurve.ipynb) | Encode a light curve to the 1536-d latent and re-express it in the global PCA basis. |
| [`03_explore_the_encodings.ipynb`](03_explore_the_encodings.ipynb) | Load the bundled PCA encoding + UMAP for every released light curve; plot the latent space and find nearest neighbours. |
| [`04_predict_flux.ipynb`](04_predict_flux.ipynb) | Forecast flux with the model's flow head and plot the p16–p84 uncertainty band. |
| [`05_the_pls_projection.ipynb`](05_the_pls_projection.ipynb) | The supervised PLS projection: the scaling, centering and matrix that take the 1536-d latent to age-covarying components, and how many to keep. |

Start at notebook 1 if you are bringing your own light curve; start at notebook 2 if you
already have prepared arrays or just want to see what the encoder does.

## Running them

Install EncoTESS with the plotting/notebook extras, then launch Jupyter:

```bash
pip install -e ".[tutorials]"
jupyter lab   # or: jupyter notebook
```

Notebook 1 needs `lightkurve` (included in `[tutorials]`, or on its own via
`pip install -e ".[preprocess]"`) and downloads from MAST. Notebooks 3 and 4 use
`matplotlib` for the figures; notebook 2 needs only the core dependencies.

For terse, copy-pasteable snippets rather than a guided walkthrough, see `../examples/`.
