"""
iskay_data_loader.py
====================

Data acquisition for PIR-kSZ pilot. Three modes:

  - "iskay" / "iskay2"  : run the upstream pairwise kSZ pipeline and
                          extract p(r), σ(r), and a ΛCDM template.
                          Requires github.com/patogallardo/iskay (or
                          iskay2; arXiv:2510.20715) to be installed and
                          the underlying ACT × SDSS data products to be
                          on disk.
  - "synthetic"         : generate a realistic test dataset from a known
                          template + Gaussian noise. Used to validate
                          the pipeline before real data is plugged in.
  - "digitized"         : load a CSV of digitized published values
                          (fallback if iskay cannot be installed).

Returns a populated KSZData container.

Design notes
------------
The iskay and digitized paths are intentionally STUBS in the pilot
release. They will be filled in once we either (a) get iskay running
locally with the ACT DR6 / SDSS DR15 data products in place, or (b)
extract the p(r) values from the Gallardo et al. PRL figure or
supplement. The synthetic path is fully functional and is what we
use to validate pir_ksz.py before real data lands.

Provenance is recorded in KSZData.metadata for every code path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

import numpy as np

from physics_engine.applications.pir_ksz import KSZData


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load(source: str, config: dict) -> KSZData:
    """
    Dispatch to the appropriate loader.

    Parameters
    ----------
    source : {"iskay", "iskay2", "synthetic", "digitized"}
    config : dict
        Source-specific.
    """
    source = source.lower()
    if source in {"iskay", "iskay2"}:
        return _load_via_iskay(config, version=source)
    elif source == "synthetic":
        return _load_synthetic(config)
    elif source == "digitized":
        return _load_digitized(config)
    else:
        raise ValueError(
            f"Unknown source {source!r}. Expected one of "
            "{'iskay', 'iskay2', 'synthetic', 'digitized'}."
        )


# ---------------------------------------------------------------------------
# Path 1: iskay / iskay2
# ---------------------------------------------------------------------------

def _load_via_iskay(config: dict, version: str = "iskay2") -> KSZData:
    """
    Run the upstream pairwise kSZ pipeline.

    Expected config keys
    --------------------
    cmb_map_path : str
        Path to ACT DR6 component-separated CMB map FITS file.
    catalog_path : str
        Path to SDSS DR15 LRG catalog (or BOSS DR11 LSS catalog for
        Calafut et al. 2021 reproduction).
    redshift_min, redshift_max : float
        Default 0.44, 0.66 (matches Gallardo et al. 2026).
    r_bins_mpc : array-like
        Default np.arange(30, 240, 10) — bin edges, 21 bins of 10 Mpc.
    aperture_disk_arcmin, aperture_ring_arcmin : float
        AP photometry geometry. Defaults match Calafut et al. 2021.
    template_path : str
        Path to a precomputed ΛCDM p(r) curve. If absent, we fall back
        to fitting an analytic surrogate to the published Gallardo
        best-fit and warn loudly.

    Returns
    -------
    KSZData

    Status: STUB.
    -------------
    The iskay run is non-trivial and we will not include the actual
    invocation here until the local environment is set up. This
    function will currently raise NotImplementedError with a clear
    message pointing to the integration steps.
    """
    try:
        # Lazy import — iskay is heavy, has fortran deps, may not be installed
        import iskay  # noqa: F401
    except ImportError as e:
        raise NotImplementedError(
            "iskay/iskay2 path requires the upstream package to be "
            "installed. To proceed:\n"
            "  1. git clone https://github.com/patogallardo/iskay\n"
            "  2. pip install -e ./iskay\n"
            "  3. Acquire ACT DR6 + SDSS DR15 data products (paths in "
            "config['cmb_map_path'] and config['catalog_path']).\n"
            "  4. Re-run with source='iskay'.\n\n"
            "For pilot validation, use source='synthetic' instead.\n"
            f"Original ImportError: {e}"
        )

    raise NotImplementedError(
        "iskay loader implementation pending. Stub raises until we "
        "have the upstream pipeline running locally. The interface "
        "above documents the expected config; pir_ksz.py is fully "
        "validated against synthetic data and will accept the "
        "iskay-produced KSZData object as soon as this stub is filled in."
    )


# ---------------------------------------------------------------------------
# Path 2: synthetic
# ---------------------------------------------------------------------------

def _load_synthetic(config: dict) -> KSZData:
    """
    Generate a realistic kSZ-like dataset for pipeline validation.

    Default parameters approximate the Gallardo et al. 2026 measurement:
      - 21 bins from 30 to 230 Mpc, 10 Mpc wide
      - Template: power law p_template ∝ -A · (r/r0)^(-0.5) reflecting
        the typical pairwise-momentum decline with separation. NOTE:
        this is NOT the true ΛCDM curve; it is a phenomenological
        stand-in adequate for pipeline testing. The real loader uses a
        ΛCDM p(r) computed via a velocity-correlation kernel.
      - Noise: Gaussian, σ ∝ 1/sqrt(N_pairs(r)) ∝ r^(-1) — bins at
        larger r have more pairs and smaller error bars (this is the
        typical scaling for pairwise estimators on a fixed catalog).
      - Amplitude calibrated so SNR per bin ≈ 5 at r ≈ 100 Mpc, similar
        to Calafut et al. 2021 quoted SNR.

    The synthetic data inject NO residual structure beyond the template
    by default. To test sensitivity, set config["inject_yukawa"] = True
    or config["inject_log"] = True; this is used in the pipeline
    sensitivity test (see tests/test_pir_ksz_synthetic.py).

    Config keys
    -----------
    n_bins : int, default 21
    r_min, r_max : float, default 30.0, 230.0 (Mpc)
    template_amplitude : float, default 1.0 (arbitrary units)
    template_power : float, default -0.5
    snr_at_100mpc : float, default 5.0
    seed : int, default 0
    inject_yukawa : bool, default False
    inject_yukawa_amplitude : float, default 0.3 (in template-amplitude units)
    inject_yukawa_scale : float, default 50.0 (Mpc)
    """
    n_bins = config.get("n_bins", 21)
    r_min = config.get("r_min", 30.0)
    r_max = config.get("r_max", 230.0)
    A = config.get("template_amplitude", 1.0)
    p = config.get("template_power", -0.5)
    snr_100 = config.get("snr_at_100mpc", 5.0)
    seed = config.get("seed", 0)
    inject_yukawa = config.get("inject_yukawa", False)
    A_y = config.get("inject_yukawa_amplitude", 0.3)
    r_d = config.get("inject_yukawa_scale", 50.0)

    rng = np.random.default_rng(seed)
    r = np.linspace(r_min, r_max, n_bins)

    # Template: phenomenological power-law decline (negative because
    # pairs fall toward each other → negative pairwise momentum sign
    # convention in the kSZ literature).
    r0 = 100.0
    p_template = -A * (r / r0) ** p

    # Noise: σ ∝ 1/r (more pairs at larger separation)
    sigma_at_100 = abs(p_template[np.argmin(np.abs(r - r0))]) / snr_100
    sigma_p = sigma_at_100 * (r0 / r)

    # Optional injection
    p_truth = p_template.copy()
    if inject_yukawa:
        # Yukawa-like additive correction: A_y · (r0/r) · exp(-r/r_d)
        # (negative sign chosen to make it a coherent perturbation
        # to the template's negative values)
        p_truth = p_template - A_y * (r0 / r) * np.exp(-r / r_d)

    p_obs = p_truth + rng.normal(0.0, sigma_p)

    metadata = {
        "source": "synthetic",
        "synthetic_config": {
            "n_bins": n_bins,
            "r_min_mpc": r_min,
            "r_max_mpc": r_max,
            "template_amplitude": A,
            "template_power": p,
            "snr_at_100mpc": snr_100,
            "seed": seed,
            "inject_yukawa": inject_yukawa,
            "inject_yukawa_amplitude": A_y if inject_yukawa else None,
            "inject_yukawa_scale": r_d if inject_yukawa else None,
        },
        "note": "Synthetic data is NOT the Gallardo et al. measurement. "
                "Used solely for pipeline validation.",
    }
    return KSZData(r=r, p_obs=p_obs, sigma_p=sigma_p,
                   p_template=p_template, metadata=metadata)


# ---------------------------------------------------------------------------
# Path 3: digitized CSV fallback
# ---------------------------------------------------------------------------

def _load_digitized(config: dict) -> KSZData:
    """
    Load a CSV of digitized published p(r) values.

    Expected CSV columns:
        r_mpc, p_obs, sigma_p, p_template

    Provenance must be recorded explicitly in metadata: which figure of
    which paper, who digitized it, with what tool, and the date. This
    path is a fallback only — for the final paper, the iskay path is
    required.

    Config keys
    -----------
    csv_path : str (required)
    provenance : dict (required)
        Free-form record describing the digitization. At minimum:
        {"paper": "...", "figure": "Fig 2a", "digitizer": "...",
         "tool": "WebPlotDigitizer 4.x", "date": "2026-04-..."}
    """
    csv_path = Path(config["csv_path"])
    provenance = config.get("provenance", {})
    if not provenance:
        warnings.warn(
            "Digitized data loaded without provenance metadata. This is "
            "acceptable for testing but MUST be filled in before the "
            "data is used in any published analysis."
        )

    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    r = arr["r_mpc"].astype(float)
    p_obs = arr["p_obs"].astype(float)
    sigma_p = arr["sigma_p"].astype(float)
    p_template = arr["p_template"].astype(float)

    metadata = {
        "source": "digitized",
        "csv_path": str(csv_path),
        "provenance": provenance,
        "warning": "Digitized values have additional uncertainty beyond "
                   "the published σ_p; consider inflating sigma_p by "
                   "5-10% if the digitization tool quoted a comparable "
                   "extraction error.",
    }
    return KSZData(r=r, p_obs=p_obs, sigma_p=sigma_p,
                   p_template=p_template, metadata=metadata)
