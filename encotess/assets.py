"""Locating bundled artifacts and fetching the large external latent banks.

Small artifacts (encoder weights, full-rank global PCA, the top-64 PCA preview, the
2-D UMAP embedding, metadata CSVs) ship inside the package under ``weights/`` and
``data/``. The full 1536-d PCA encoding of every light curve is too large for a wheel
(~400 MB) and is downloaded on demand from an external host (HuggingFace) into a
local cache.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_WEIGHTS = _PKG_DIR / 'weights'
_DATA = _PKG_DIR / 'data'


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled artifact not found: {path}\n"
            "This usually means an incomplete or corrupted install; try reinstalling "
            "the package.")
    return path


# ---- bundled (small) artifacts -------------------------------------------
def encoder_weights_path() -> Path:
    return _require(_WEIGHTS / 'encotess_weights.pt')


def pca_weights_path() -> Path:
    """Full-rank (1536-comp) unwhitened global-PCA artifact."""
    return _require(_WEIGHTS / 'global_pca.npz')


def pca_preview_path() -> Path:
    """Bundled compact PCA encoding (top-64) for every released light curve."""
    return _require(_DATA / 'latents_pca64.npz')


def umap_path() -> Path:
    """Bundled 2-D UMAP embedding for every released light curve."""
    return _require(_DATA / 'umap.npz')


# ---- large external artifacts (HuggingFace) ------------------------------
# Downloaded on demand into the local cache; URLs/checksums are filled in once
# uploaded. Until then the download helpers raise an informative error.
#   - the full 1536-d PCA encoding (~400 MB)
#   - the per-light-curve metadata CSVs (per-sector ~14 MB, per-star ~4.5 MB)
PCA_FULL = {'url': None, 'sha256': None, 'filename': 'encodings_pca_full.npz'}
METADATA = {
    'per_sector': {'url': None, 'sha256': None, 'filename': 'metadata_per_sector.csv'},
    'per_star':   {'url': None, 'sha256': None, 'filename': 'metadata_per_star.csv'},
}


def cache_dir() -> Path:
    """Local cache directory for downloaded latent banks."""
    root = os.environ.get('ENCOTESS_CACHE',
                          str(Path.home() / '.cache' / 'encotess'))
    d = Path(root)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _download(entry: dict, human: str, force: bool = False) -> Path:
    """Fetch a registry entry ({url, sha256, filename}) into the local cache."""
    dest = cache_dir() / entry['filename']
    if dest.exists() and not force:
        if entry['sha256'] and _sha256(dest) != entry['sha256']:
            raise IOError(f"Cached {dest} failed checksum; re-run with force=True.")
        return dest
    if not entry['url']:
        raise NotImplementedError(
            f"The download URL for the {human} is not configured yet. Once hosted on "
            "HuggingFace, set its url/sha256 in encotess/assets.py.")
    import urllib.request
    tmp = dest.with_suffix(dest.suffix + '.part')
    urllib.request.urlretrieve(entry['url'], tmp)
    if entry['sha256'] and _sha256(tmp) != entry['sha256']:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Downloaded {human} failed checksum verification.")
    tmp.replace(dest)
    return dest


def download_latents_pca(force: bool = False) -> Path:
    """Fetch the full 1536-d PCA encoding (~400 MB) to the local cache.

    Hosted on HuggingFace. Until the URL is configured, use the bundled top-64
    preview (``encotess.assets.pca_preview_path()``).
    """
    return _download(PCA_FULL, 'full 1536-d PCA encoding', force)


def download_metadata(which: str = 'per_sector', force: bool = False) -> Path:
    """Fetch a per-light-curve metadata CSV ('per_sector' | 'per_star') to the cache.

    Hosted on HuggingFace alongside the full PCA encoding.
    """
    if which not in METADATA:
        raise KeyError(f"Unknown metadata {which!r}; choose from {list(METADATA)}.")
    return _download(METADATA[which], f"{which} metadata CSV", force)
