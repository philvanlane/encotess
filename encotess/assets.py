"""Locating bundled artifacts and fetching the large external PCA encoding.

Nearly everything ships inside the package under ``weights/`` and ``data/``: the
encoder weights, the full-rank global PCA, the top-64 PCA preview, the 2-D UMAP
embedding, the per-star PLS encodings, and every metadata CSV. The one exception is
the full 1536-d PCA encoding of every light curve, which is too large for a wheel
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


def pls_encoding_path(n_components: int = 3) -> Path:
    """Bundled per-star PLS encoding (age-supervised projection of the latent).

    ``n_components=3``: fit on the 2,893-star literature-Prot subset — reproduces the
      paper's ``latent_pls3`` age model.
    ``n_components=16``: fit on all 9,221 age-labelled stars (broader; median-age target).

    Keys: ``gaia_ids``, ``bank`` (FGKMcal/hosts/thickdisk), ``pls`` (N, n_components),
    and ``in_age_fit`` (whether the star was in the fit population). See DATASET.md.
    """
    if n_components == 3:
        return _require(_DATA / 'encodings_pls3_star.npz')
    if n_components == 16:
        return _require(_DATA / 'encodings_pls16_star.npz')
    raise ValueError(f"No PLS encoding with n_components={n_components} (have 3, 16).")


_METADATA_FILES = {
    'sector':         'metadata_sector.csv',
    'FGKMcal_star':   'metadata_FGKMcal_star.csv',
    'hosts_star':     'metadata_hosts_star.csv',
    'thickdisk_star': 'metadata_thickdisk_star.csv',
}


def metadata_path(which: str = 'sector') -> Path:
    """Path to a bundled metadata CSV.

    ``which`` is one of: 'sector' (one row per light curve, row-aligned to the
    encodings), 'FGKMcal_star', 'hosts_star', 'thickdisk_star' (one row per star).
    See DATASET.md for the column dictionary and provenance.
    """
    if which not in _METADATA_FILES:
        raise KeyError(
            f"Unknown metadata {which!r}; choose from {list(_METADATA_FILES)}.")
    return _require(_DATA / _METADATA_FILES[which])


# ---- large external artifact (HuggingFace) -------------------------------
# The full 1536-d PCA encoding is downloaded on demand into the local cache.
# (Array key inside the npz: 'latents_pca'.) Hosted as a public HuggingFace dataset;
# the sha256 is the file's git-LFS object id, verified against the uploaded file.
PCA_FULL = {
    'url': 'https://huggingface.co/datasets/philvanlane/encotess/resolve/main/encodings_pca_full.npz',
    'sha256': '2edf5a73c67471b5cb83c9470b65a86d053d5078252a35c164ec40d1014d0e7f',
    'filename': 'encodings_pca_full.npz',
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
