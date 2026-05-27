"""
pir_ksz.py
==========

PIR-JEPA application: kinematic Sunyaev-Zel'dovich (kSZ) pairwise-momentum
residual analysis.

Goal
----
Apply the PIR rank-1 / rank-2 / Δs framework to the cluster pairwise kSZ
measurement of Gallardo et al. 2026 (Phys. Rev. Lett. 136, 151002,
DOI:10.1103/rk8v-rcm3) over separations r ∈ [30, 230] Mpc.

Design discipline
-----------------
PIR-kSZ is purely ADDITIVE. It calls discover_law() with the EXACT
signature it has today and reads the metadata fields the function
already exposes. No source-file modification to symbolic_search.py,
jepa_langevin.py, or any other core file. PIR-Bench v3.1, SPARC v3,
LIGO results all remain byte-identical to canonical Zenodo records.

The simplify-timeout monkeypatch
---------------------------------
Empirically we observed sympy.simplify hanging indefinitely (>2700 s
per call) on candidate expressions derived from noisy kSZ residual
data. The hang originates in trigsimp → factor → integer-factorization
on huge rationals introduced by tiny float coefficients (~1e-9). This
pathology does NOT affect clean PIR-Bench/SPARC/LIGO data because
those produce expressions with integer/rational coefficients that
simplify in microseconds.

To isolate the fix to PIR-kSZ:
  - We monkeypatch sympy.simplify INSIDE PIR-kSZ entry points only,
    via a context manager (`_simplify_timeout_patch`).
  - The patched simplify runs the original in a daemon thread with a
    timeout (default 10 s); on timeout it returns the unsimplified
    expression (which lambdifies identically — simplify is cosmetic).
  - The patch is REVERTED on exit. Other PIR applications run with the
    unmodified sympy.simplify.

This decouples PIR-kSZ from core entirely. We never edit
symbolic_search.py.

How the analysis works
----------------------
  1. Pre-subtract the ΛCDM template:    residual = p_obs - p_template
  2. Write (r, residual) to a temp CSV
  3. Call discover_law(csv_path=tmp, target_var="residual",
                       return_metadata=True, use_jepa=True, gamma=0.2,
                       langevin_steps=500, ...) — under simplify patch
  4. Read delta_s, rank2_expr, jepa_combined_score from structure_metadata
  5. Bootstrap the noise floor under H₀ by drawing p_obs from p_template
     + Gaussian(σ_p) and repeating steps 1-4 many times

Predicted outcome (publishable null)
------------------------------------
|Δs_obs| ≤ Δs_noise_95 → no residual symbolic correction detected; the
kSZ pairwise momentum is consistent with the ΛCDM template at present
statistical precision.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import sympy as sp

from physics_engine.discovery.symbolic_search import discover_law

# ---------------------------------------------------------------------------
# Project design constants — PINNED to AGENT_INSTRUCTIONS.md
# ---------------------------------------------------------------------------
GAMMA_JEPA = 0.2
ALPHA_OT = 0.7
BETA_OT = 0.3
LANGEVIN_STEPS_DEFAULT = 500
NUMERICAL_EQUIV_TOL = 0.02

# kSZ-specific: simplify timeout. Healthy expressions complete in <0.1 s;
# we set a generous 10 s ceiling so any hang is terminated long before
# it dominates wall time.
SIMPLIFY_TIMEOUT_SECONDS = 10.0


# ---------------------------------------------------------------------------
# sympy.simplify timeout patch — context manager, applies inside PIR-kSZ only
# ---------------------------------------------------------------------------

def _simplify_with_timeout(expr, timeout: float = SIMPLIFY_TIMEOUT_SECONDS,
                           original_simplify=None, **kwargs):
    """
    Run sympy.simplify in a daemon thread with a timeout.

    On timeout, return the input expression unsimplified (semantically
    identical for lambdify; only cosmetically different).

    Note: a daemon thread cannot be hard-killed in Python. The runaway
    thread continues consuming CPU until the program exits or its
    recursive sympy call returns. Acceptable for a bootstrap loop where
    the main thread proceeds with the unsimplified expression
    immediately and the runaway thread is simply ignored. Each draw
    leaks at most one such thread; daemon means they all die at process
    exit.
    """
    if original_simplify is None:
        # Should not happen — defensive
        return expr

    holder: dict = {"result": expr, "completed": False, "error": None}

    def _runner():
        try:
            holder["result"] = original_simplify(expr, **kwargs)
            holder["completed"] = True
        except Exception as e:
            holder["error"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if not holder["completed"]:
        # Timed out — fall through with unsimplified expression.
        # The runaway thread continues but we ignore it (daemon).
        return expr
    if holder["error"] is not None:
        # simplify raised — return unsimplified rather than propagating.
        # discover_law's downstream code is robust to either form.
        return expr
    return holder["result"]


@contextmanager
def _simplify_timeout_patch(timeout: float = SIMPLIFY_TIMEOUT_SECONDS):
    """
    Monkeypatch sympy.simplify to use the timeout-guarded version.
    Reverts on exit. Thread-safety: this is process-global; do NOT call
    SPARC/LIGO/PIR-Bench from inside a PIR-kSZ run.
    """
    original = sp.simplify

    def _patched(expr, *args, **kwargs):
        return _simplify_with_timeout(
            expr, timeout=timeout, original_simplify=original, **kwargs
        )

    sp.simplify = _patched
    # Also patch the module-internal alias if present in symbolic_search's
    # namespace, since `sp.simplify(...)` there resolves through the same
    # sympy module attribute (which we've just replaced).
    try:
        yield
    finally:
        sp.simplify = original


# ---------------------------------------------------------------------------
# Compatibility assertion — runs once at module import time
# ---------------------------------------------------------------------------

def _assert_compatible_discover_law() -> None:
    """Sanity-check that discover_law has the parameters PIR-kSZ depends on."""
    import inspect
    sig = inspect.signature(discover_law)
    required = {
        "csv_path", "target_var", "return_metadata",
        "use_jepa", "gamma", "langevin_steps",
        "use_ot_loss", "alpha", "beta",
        "use_residual", "use_ransac", "use_sparse",
        "enforce_dimensions", "random_state",
    }
    missing = required - set(sig.parameters.keys())
    if missing:
        raise RuntimeError(
            f"discover_law signature has drifted. PIR-kSZ requires "
            f"these parameters but they are missing: {sorted(missing)}. "
            f"Coordinate with symbolic_search.py before re-running."
        )


_assert_compatible_discover_law()


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class KSZData:
    """
    Standardized container for pairwise-kSZ measurements.

    r          — comoving separation bin centers (Mpc), shape (n_bins,)
    p_obs      — observed pairwise momentum, shape (n_bins,)
    sigma_p    — 1σ statistical uncertainty per bin, shape (n_bins,)
    p_template — ΛCDM-templated p(r) on same grid, shape (n_bins,)
    metadata   — provenance dict
    """
    r: np.ndarray
    p_obs: np.ndarray
    sigma_p: np.ndarray
    p_template: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.r)
        for name, arr in [("p_obs", self.p_obs),
                          ("sigma_p", self.sigma_p),
                          ("p_template", self.p_template)]:
            if len(arr) != n:
                raise ValueError(
                    f"KSZData length mismatch: r has {n} bins but "
                    f"{name} has {len(arr)}"
                )
        if np.any(self.sigma_p <= 0):
            raise ValueError("sigma_p must be strictly positive")
        if np.any(self.r <= 0):
            raise ValueError("r must be strictly positive (Mpc)")

    @property
    def residual(self) -> np.ndarray:
        return self.p_obs - self.p_template

    @property
    def n_bins(self) -> int:
        return len(self.r)


# ---------------------------------------------------------------------------
# Data loader (delegates)
# ---------------------------------------------------------------------------

def load_ksz_data(source: str = "synthetic",
                  config: Optional[dict] = None) -> KSZData:
    """source : {'iskay', 'iskay2', 'synthetic', 'digitized'}"""
    from physics_engine.applications import iskay_data_loader as loader
    return loader.load(source=source, config=config or {})


# ---------------------------------------------------------------------------
# Core: rank-2 search via canonical discover_law
# ---------------------------------------------------------------------------

def discover_rank2_via_residual(
    data: KSZData,
    *,
    use_jepa: bool = True,
    use_ot_loss: bool = True,
    use_ransac: bool = True,
    use_sparse: bool = True,
    use_residual_flag: bool = True,
    enforce_dimensions: bool = False,
    gamma: float = GAMMA_JEPA,
    alpha: float = ALPHA_OT,
    beta: float = BETA_OT,
    langevin_steps: int = LANGEVIN_STEPS_DEFAULT,
    random_state: int = 42,
    task_name: str = "pir_ksz_residual",
    tmp_dir: Optional[Path] = None,
    simplify_timeout: float = SIMPLIFY_TIMEOUT_SECONDS,
) -> dict:
    """
    Discover a symbolic correction to the ΛCDM template via discover_law.

    All discover_law calls run under the simplify-timeout patch so that
    pathological candidate expressions cannot hang the bootstrap.
    """
    df = pd.DataFrame({
        "r": data.r,
        "residual": data.residual,
    })

    if tmp_dir is None:
        tmp_dir = Path(tempfile.gettempdir())
    else:
        tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / f"pir_ksz_{task_name}_{int(time.time()*1000)}.csv"
    df.to_csv(csv_path, index=False)

    try:
        with _simplify_timeout_patch(timeout=simplify_timeout):
            result = discover_law(
                csv_path=str(csv_path),
                target_var="residual",
                use_jepa=use_jepa,
                use_ot_loss=use_ot_loss,
                use_ransac=use_ransac,
                use_residual=use_residual_flag,
                use_sparse=use_sparse,
                enforce_dimensions=enforce_dimensions,
                gamma=gamma,
                alpha=alpha,
                beta=beta,
                langevin_steps=langevin_steps,
                random_state=random_state,
                task_name=task_name,
                return_metadata=True,
            )
    finally:
        try:
            csv_path.unlink()
        except OSError:
            pass

    best_expr, val_error, significant, structure_metadata = result

    return {
        "best_expr": best_expr,
        "rank2_expr": structure_metadata.get("rank2_expr"),
        "delta_s": structure_metadata.get("delta_s"),
        "jepa_combined_score": structure_metadata.get("jepa_combined_score"),
        "validation_error": float(val_error),
        "significant": significant,
        "structure_metadata": structure_metadata,
    }


# ---------------------------------------------------------------------------
# Bootstrap noise floor under H₀
# ---------------------------------------------------------------------------

def bootstrap_delta_s_null(
    data: KSZData,
    *,
    n_bootstrap: int = 200,
    seed: int = 0,
    verbose: bool = True,
    tmp_dir: Optional[Path] = None,
    per_call_timeout: Optional[float] = 120.0,
    **discover_kwargs,
) -> dict:
    """
    Bootstrap distribution of Δs under H₀ (no residual structure).

    per_call_timeout
        Wall-clock ceiling per bootstrap draw, in seconds. If a single
        discover_law call exceeds this (despite the simplify patch),
        the draw is recorded as a failure and we move on. This is a
        belt-and-braces guard; in practice the simplify patch alone is
        usually enough. Set to None to disable.

    Returns
    -------
    dict
        delta_s_null, delta_s_noise_95, delta_s_noise_99,
        n_successes, n_failures, wall_seconds, draw_durations
    """
    rng = np.random.default_rng(seed)
    delta_s_draws: list[float] = []
    n_failures = 0
    draw_durations: list[float] = []
    t0 = time.time()

    for i in range(n_bootstrap):
        p_obs_null = data.p_template + rng.normal(0.0, data.sigma_p)
        null_data = KSZData(
            r=data.r.copy(),
            p_obs=p_obs_null,
            sigma_p=data.sigma_p.copy(),
            p_template=data.p_template.copy(),
            metadata={**data.metadata, "bootstrap_draw": i},
        )

        draw_t0 = time.time()
        try:
            result = discover_rank2_via_residual(
                null_data,
                random_state=int(rng.integers(0, 2**31 - 1)),
                task_name=f"pir_ksz_bootstrap_{i:04d}",
                tmp_dir=tmp_dir,
                **discover_kwargs,
            )
            draw_dt = time.time() - draw_t0
            draw_durations.append(draw_dt)

            if per_call_timeout is not None and draw_dt > per_call_timeout:
                # The simplify patch let it finish but it was still slow.
                # Record but do NOT count this in the noise distribution
                # — a draw that took 5 minutes is not representative.
                n_failures += 1
                if verbose:
                    print(f"  bootstrap {i+1}: SLOW ({draw_dt:.1f}s > "
                          f"{per_call_timeout}s) — discarded")
                continue

            ds = result["delta_s"]
            if ds is None:
                n_failures += 1
                if verbose:
                    print(f"  bootstrap {i+1}: Δs=None (no rank-2)  "
                          f"({draw_dt:.1f}s)")
            else:
                delta_s_draws.append(float(ds))
                if verbose:
                    print(f"  bootstrap {i+1}: Δs={float(ds):+.4f}  "
                          f"({draw_dt:.1f}s)")
        except Exception as e:
            draw_dt = time.time() - draw_t0
            draw_durations.append(draw_dt)
            n_failures += 1
            if verbose:
                print(f"  bootstrap {i+1}: FAILED ({type(e).__name__}: {e})"
                      f"  ({draw_dt:.1f}s)")

        if verbose and (i + 1) % max(1, n_bootstrap // 10) == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_bootstrap - i - 1)
            print(f"  --- {i+1}/{n_bootstrap}  successes={len(delta_s_draws)}"
                  f"  failures={n_failures}  elapsed={elapsed:.0f}s  "
                  f"eta={eta:.0f}s")

    if len(delta_s_draws) == 0:
        raise RuntimeError(
            f"All {n_bootstrap} bootstrap draws failed to produce a Δs. "
            "Check discover_law's behavior on the residual data; the "
            "simplify-timeout patch may need tuning, or the data may be "
            "too noise-limited for any rank-2 candidate to emerge."
        )

    arr = np.asarray(delta_s_draws)
    abs_arr = np.abs(arr)
    return {
        "delta_s_null": arr,
        "delta_s_noise_95": float(np.percentile(abs_arr, 95)),
        "delta_s_noise_99": float(np.percentile(abs_arr, 99)),
        "n_successes": len(delta_s_draws),
        "n_failures": n_failures,
        "draw_durations": draw_durations,
        "wall_seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_pir_ksz_pilot(
    data: KSZData,
    *,
    n_bootstrap: int = 200,
    output_dir: Optional[Path] = None,
    seed: int = 0,
    per_call_timeout: Optional[float] = 120.0,
    **discover_kwargs,
) -> dict:
    """Full PIR-kSZ pipeline with auditable JSON report."""
    t0 = time.time()
    print("=" * 68)
    print("PIR-kSZ pilot: pairwise momentum residual analysis")
    print("=" * 68)
    print(f"  data: n_bins={data.n_bins}, "
          f"r range = [{data.r.min():.1f}, {data.r.max():.1f}] Mpc")
    print(f"  source: {data.metadata.get('source', 'unknown')}")
    print(f"  simplify_timeout = {SIMPLIFY_TIMEOUT_SECONDS:.1f}s, "
          f"per_call_timeout = {per_call_timeout}s")
    print()

    # Stage 1
    print("[1/3] Searching rank-2 symbolic correction on real residual...")
    stage1_t0 = time.time()
    obs_result = discover_rank2_via_residual(
        data,
        random_state=seed,
        task_name="pir_ksz_observed",
        **discover_kwargs,
    )
    stage1_dt = time.time() - stage1_t0
    delta_s_obs = obs_result["delta_s"]
    if delta_s_obs is None:
        print("  Δs is None on real data (no rank-2 candidate produced). "
              "Treating |Δs_obs|=0 in verdict; report records None.")
        delta_s_obs_eff = 0.0
    else:
        delta_s_obs_eff = float(delta_s_obs)
    print(f"      best correction  = {obs_result['best_expr']}")
    print(f"      rank2 correction = {obs_result['rank2_expr']}")
    print(f"      Δs_obs           = {delta_s_obs_eff:+.4f}"
          f"{'' if delta_s_obs is not None else ' (None → 0 in verdict)'}")
    print(f"      validation_error = {obs_result['validation_error']:.4f}")
    print(f"      stage 1 wall     = {stage1_dt:.1f}s")
    print()

    # Stage 2
    print(f"[2/3] Bootstrapping Δs noise floor under H₀ (n={n_bootstrap})...")
    boot = bootstrap_delta_s_null(
        data,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
        verbose=True,
        per_call_timeout=per_call_timeout,
        **discover_kwargs,
    )
    print(f"      Δs_noise (95%)   = {boot['delta_s_noise_95']:.4f}")
    print(f"      Δs_noise (99%)   = {boot['delta_s_noise_99']:.4f}")
    print(f"      successes/total  = {boot['n_successes']}/{n_bootstrap}")
    print(f"      bootstrap wall   = {boot['wall_seconds']:.1f}s")
    if boot["draw_durations"]:
        durs = np.asarray(boot["draw_durations"])
        print(f"      draw timing      = "
              f"min={durs.min():.1f}s, "
              f"median={np.median(durs):.1f}s, "
              f"max={durs.max():.1f}s")
    print()

    # Stage 3
    consistent = abs(delta_s_obs_eff) <= boot["delta_s_noise_95"]
    print("[3/3] Verdict")
    headline = ("NULL CONFIRMED — no residual structure detected"
                if consistent
                else "NULL REJECTED — residual structure flagged")
    print(f"      >>> {headline} <<<")
    print(f"      |Δs_obs|={abs(delta_s_obs_eff):.4f}  vs  "
          f"Δs_noise_95={boot['delta_s_noise_95']:.4f}")
    print()

    report = {
        "rank1_correction_expr": str(obs_result["best_expr"]),
        "rank2_correction_expr": str(obs_result["rank2_expr"]),
        "delta_s_obs": delta_s_obs_eff,
        "delta_s_obs_raw": (None if delta_s_obs is None
                            else float(delta_s_obs)),
        "delta_s_noise_95": boot["delta_s_noise_95"],
        "delta_s_noise_99": boot["delta_s_noise_99"],
        "consistent_with_null": bool(consistent),
        "validation_error_obs": obs_result["validation_error"],
        "n_bootstrap": n_bootstrap,
        "bootstrap_n_successes": boot["n_successes"],
        "bootstrap_n_failures": boot["n_failures"],
        "stage1_wall_seconds": stage1_dt,
        "wall_seconds": time.time() - t0,
        "metadata": {
            **data.metadata,
            "seed": seed,
            "simplify_timeout_seconds": SIMPLIFY_TIMEOUT_SECONDS,
            "per_call_timeout_seconds": per_call_timeout,
            "discover_kwargs": {k: str(v) for k, v in discover_kwargs.items()},
            "design_constants": {
                "gamma_jepa": GAMMA_JEPA,
                "alpha_ot": ALPHA_OT,
                "beta_ot": BETA_OT,
                "langevin_steps_default": LANGEVIN_STEPS_DEFAULT,
            },
        },
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "pir_ksz_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        np.savez(
            output_dir / "pir_ksz_data.npz",
            r=data.r,
            p_obs=data.p_obs,
            sigma_p=data.sigma_p,
            p_template=data.p_template,
        )
        np.save(output_dir / "pir_ksz_bootstrap.npy", boot["delta_s_null"])
        if boot["draw_durations"]:
            np.save(output_dir / "pir_ksz_draw_durations.npy",
                    np.asarray(boot["draw_durations"]))

        with open(output_dir / "_provenance.txt", "w") as f:
            f.write(
                "PIR-kSZ pilot run provenance\n"
                "============================\n"
                f"timestamp:                {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"source:                   {data.metadata.get('source', 'unknown')}\n"
                f"n_bins:                   {data.n_bins}\n"
                f"n_bootstrap:              {n_bootstrap}\n"
                f"seed:                     {seed}\n"
                f"simplify_timeout_seconds: {SIMPLIFY_TIMEOUT_SECONDS}\n"
                f"per_call_timeout_seconds: {per_call_timeout}\n"
                f"discover_kwargs:          {discover_kwargs}\n"
                "feeds: Section 4 (real-data Δs), Section 5 "
                "(bootstrap noise floor) of PIR-kSZ pilot paper.\n"
            )

        print(f"  saved to {output_dir}/")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PIR-kSZ pilot driver")
    parser.add_argument("--source", default="synthetic",
                        choices=["iskay", "iskay2", "synthetic", "digitized"])
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="./ksz_results/quick")
    parser.add_argument("--per-call-timeout", type=float, default=120.0)
    args = parser.parse_args()

    data = load_ksz_data(source=args.source)
    run_pir_ksz_pilot(
        data,
        n_bootstrap=args.n_bootstrap,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        per_call_timeout=args.per_call_timeout,
    )
