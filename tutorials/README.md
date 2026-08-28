# Tutorials

Step-by-step notebooks that walk through what EncoTESS does. Each one is short and
runs top-to-bottom on the bundled data — no external downloads required.

| Notebook | What it covers |
|---|---|
| [`01_encode_a_lightcurve.ipynb`](01_encode_a_lightcurve.ipynb) | Encode a light curve to the 1536-d latent and re-express it in the global PCA basis. |
| [`02_explore_the_encodings.ipynb`](02_explore_the_encodings.ipynb) | Load the bundled PCA encoding + UMAP for every released light curve; plot the latent space and find nearest neighbours. |
| [`03_predict_flux.ipynb`](03_predict_flux.ipynb) | Forecast flux with the model's flow head and plot the p16–p84 uncertainty band. |

## Running them

Install EncoTESS with the plotting/notebook extras, then launch Jupyter:

```bash
pip install -e ".[tutorials]"
jupyter lab   # or: jupyter notebook
```

Notebooks 2 and 3 use `matplotlib` for the figures; notebook 1 needs only the core
dependencies.

For terse, copy-pasteable snippets rather than a guided walkthrough, see `../examples/`.
