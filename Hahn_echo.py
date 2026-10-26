# Copyright (c) 2026 Shardul Kadam
# Licensed under the MIT License.
# See the LICENSE file in the project root for details.


"""
Hahn_echo.py
================

Hahn-echo experiment for an S-T0 spin qubit under RTS-sum (1/f) charge noise,
simulated with QuTiP (>= 5). Extracts the echo time T2 from the echo decay.

Hamiltonian (hbar = 1, angular frequencies) -- same as ramsey_st0.py:

    H(t) = (1/2) * 2*pi * [ J0 + dJ/d_eps * d_eps(t) ] * sigma_x
         + (1/2) * 2*pi * dBz                          * sigma_z

with J0 = 119 MHz, dJ/d_eps = 160 MHz/mV, dBz = 42.1 MHz and d_eps(t) the
charge-noise trace from pink_noise.py (sum of random telegraph signals).

Units inside the solver: time in ns, angular frequency in rad/ns.
Public inputs/outputs use lab units (MHz, mV/uV, ns).

Echo protocol (ideal, instantaneous pulses)
-------------------------------------------
  1. pi/2 pulse:   |phi> = (|e0> + |e1>)/sqrt(2)   (equator of the noise-free H0)
  2. free evolution for T/2 under the noisy H(t)
  3. pi pulse about y at t = T/2
  4. free evolution for another T/2
  5. final pi/2 pulse and readout

A pi rotation about y maps sigma_x -> -sigma_x and sigma_z -> -sigma_z, i.e.
H -> -H, so everything that is constant in time (J0, dBz and any quasi-static
part of the noise) is refocused exactly. Only noise that changes between the
two halves survives. (A pi pulse about x would NOT refocus the J0 term.)

Observable: W(T) = | <r(T)> |, the length of the noise-averaged Bloch vector at
the end of the sequence (W = 1 without noise). T = total evolution time
(= 2 tau). The fit model is  W(T) = exp(-(T/T2)^alpha).

A single QuTiP propagator solve per noise realization gives U(0 -> t) for all
t; the echo for every total time T then follows exactly from
    U_echo(T) = U(T) U(T/2)^dagger  U_pi  U(T/2),
so the full decay curve costs about as much as one Ramsey run.

Plots and prints in the `__main__` block run only when this file is executed
directly (python hahn_echo_st0.py). Importing and calling
`hahn_echo_simulation(...)` never plots or prints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import qutip as qt
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from RTS_noise import generate_pink_noise

# ---------------------------------------------------------------------------
# Device parameters (from the paper's S-T0 Hamiltonian)
# ---------------------------------------------------------------------------

J0_MHZ = 119.0                 # exchange at the operating point
DBZ_MHZ = 42.1                 # magnetic-field gradient
DJ_DEPS_MHZ_PER_MV = 160.0     # dJ / d_eps

_UNIT_TO_MV = {"V": 1e3, "mV": 1.0, "uV": 1e-3, "µV": 1e-3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def splitting_MHz(J0_MHz: float = J0_MHZ, dBz_MHz: float = DBZ_MHZ) -> float:
    """Noise-free qubit splitting  Omega/2pi = sqrt(J0^2 + dBz^2)  [MHz]."""
    return float(np.hypot(J0_MHz, dBz_MHz))


def _sigma_longitudinal_rad_per_ns(
    noise_rms: float, noise_units: str, J0_MHz: float, dBz_MHz: float,
    dJ_dEps_MHz_per_mV: float,
) -> float:
    """RMS of the noise projected on the eigenaxis (cos(theta) = J0/Omega) [rad/ns]."""
    sigma_mV = noise_rms * _UNIT_TO_MV[noise_units]
    cos_theta = J0_MHz / splitting_MHz(J0_MHz, dBz_MHz)
    return 2.0 * np.pi * cos_theta * dJ_dEps_MHz_per_mV * sigma_mV * 1e-3


def estimate_T2star_static_ns(
    noise_rms: float, noise_units: str = "uV", J0_MHz: float = J0_MHZ,
    dBz_MHz: float = DBZ_MHZ, dJ_dEps_MHz_per_mV: float = DJ_DEPS_MHZ_PER_MV,
) -> float:
    """Quasi-static (Ramsey) estimate  T2* = sqrt(2) / sigma_longitudinal  [ns]."""
    s = _sigma_longitudinal_rad_per_ns(noise_rms, noise_units, J0_MHz, dBz_MHz, dJ_dEps_MHz_per_mV)
    return float(np.sqrt(2.0) / s)


def estimate_T2echo_1f_ns(
    noise_rms: float, noise_units: str = "uV", f_min: float = 1.0, f_max: float = 1e8,
    J0_MHz: float = J0_MHZ, dBz_MHz: float = DBZ_MHZ,
    dJ_dEps_MHz_per_mV: float = DJ_DEPS_MHZ_PER_MV,
) -> float:
    """
    Analytic echo time for ideal 1/f noise in [f_min, f_max]  [ns].

    One-sided PSD S(f) = A^2 / f with A^2 = sigma^2 / ln(f_max/f_min). The echo
    phase variance is A^2 T^2 ln 2, so W(T) = exp(-A^2 T^2 ln2 / 2) and
        T2 = sqrt(2 ln(f_max/f_min) / ln 2) / sigma_longitudinal.
    Intended as an order-of-magnitude check, not a fit target.
    """
    s = _sigma_longitudinal_rad_per_ns(noise_rms, noise_units, J0_MHz, dBz_MHz, dJ_dEps_MHz_per_mV)
    return float(np.sqrt(2.0 * np.log(f_max / f_min) / np.log(2.0)) / s)


def _stretched_exp(t: NDArray, T2: float, alpha: float) -> NDArray:
    return np.exp(-(t / T2) ** alpha)


def fit_echo(
    t_ns: NDArray[np.float64], W: NDArray[np.float64], w_fit_min: float = 0.15
) -> dict:
    """
    Fit W(T) = exp(-(T/T2)^alpha) on the contiguous points with W >= w_fit_min
    (the tail is dominated by the statistical floor ~ 0.89/sqrt(N)).
    Also returns the model-free 1/e time.
    """
    below = np.where(W < np.exp(-1.0))[0]
    if below.size:
        k = below[0]
        t_1e = float(np.interp(np.exp(-1.0), [W[k], W[k - 1]], [t_ns[k], t_ns[k - 1]]))
    else:
        t_1e = float("nan")

    mask = W >= w_fit_min
    first_bad = np.where(~mask)[0]
    if first_bad.size:
        mask[first_bad[0]:] = False
    if mask.sum() < 5:
        raise RuntimeError("Too few points above w_fit_min; increase t_max_ns or lower w_fit_min.")

    p0 = (t_1e if np.isfinite(t_1e) else float(t_ns[mask][-1]), 1.5)
    popt, pcov = curve_fit(
        _stretched_exp, t_ns[mask], W[mask], p0=p0, bounds=([1e-3, 0.3], [np.inf, 4.0])
    )
    perr = np.sqrt(np.diag(pcov))
    return {
        "T2_ns": float(popt[0]), "alpha": float(popt[1]),
        "T2_err_ns": float(perr[0]), "alpha_err": float(perr[1]),
        "T_1e_ns": t_1e, "fit_mask": mask,
    }


def _bloch(v: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Bloch vectors (x, y, z) of an array of kets with shape (n, 2)."""
    a, b = v[:, 0], v[:, 1]
    c = np.conj(a) * b
    return np.column_stack([2.0 * c.real, 2.0 * c.imag, np.abs(a) ** 2 - np.abs(b) ** 2])


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HahnEchoResult:
    t_total_ns: NDArray[np.float64]        # total evolution time T = 2*tau
    W: NDArray[np.float64]                 # echo coherence W(T)
    T2_ns: float
    alpha: float
    T2_err_ns: float
    alpha_err: float
    T_1e_ns: float
    fit_mask: NDArray[np.bool_]
    demo: list = field(default_factory=list)   # in-sequence curves (for plotting)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        p = self.params
        u = p["noise_units"].replace("uV", "µV")
        j_rms = p["noise_rms"] * _UNIT_TO_MV[p["noise_units"]] * p["dJ_dEps_MHz_per_mV"]
        return "\n".join([
            "========== Hahn-echo simulation (S-T0 qubit) ==========",
            f"J0 = {p['J0_MHz']:g} MHz, dBz = {p['dBz_MHz']:g} MHz, dJ/d_eps = {p['dJ_dEps_MHz_per_mV']:g} MHz/mV",
            f"Noise RMS            = {p['noise_rms']:g} {u}   (J-noise {j_rms:.3g} MHz)",
            f"Noise band           = {p['f_min']:g} - {p['f_max']:.3g} Hz,  realizations = {p['n_realizations']}",
            f"fs = {p['fs']:.3g} Hz,  t_max = {p['t_max_ns']:.0f} ns,  seed = {p['seed']}",
            f"T2 (stretched fit)   = {self.T2_ns:.1f} +/- {self.T2_err_ns:.1f} ns",
            f"Stretch exponent     = {self.alpha:.2f} +/- {self.alpha_err:.2f}",
            f"T2 (1/e crossing)    = {self.T_1e_ns:.1f} ns",
            f"Analytic 1/f estimate= {p['T2echo_1f_estimate_ns']:.1f} ns",
            f"Quasi-static T2*     = {p['T2star_static_estimate_ns']:.1f} ns  (Ramsey estimate)",
            f"Echo gain T2/T2*     = {self.T2_ns / p['T2star_static_estimate_ns']:.2f}",
            f"Runtime              = {p['runtime_s']:.1f} s",
            "=======================================================",
        ])

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Ensemble engine
# ---------------------------------------------------------------------------

def _run_ensemble(
    noise_rms: float, noise_units: str, J0_MHz: float, dBz_MHz: float,
    dJ_dEps_MHz_per_mV: float, f_min: float, f_max: float, fs: float,
    t_max_ns: float, n_realizations: int, n_fluctuators: int | None,
    seed: int | None, demo_total_ns: list[float], verbose: bool,
) -> dict:
    two_pi = 2.0 * np.pi
    sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
    H0 = 0.5 * two_pi * J0_MHz * 1e-3 * sx + 0.5 * two_pi * dBz_MHz * 1e-3 * sz
    Hn = 0.5 * sx                                           # multiplies noise [rad/ns]

    _, evecs = H0.eigenstates()
    phi = (evecs[0] + evecs[1]).unit().full().ravel()       # state after first pi/2
    U_pi = (-1j * np.pi / 2.0 * sy).expm().full()           # ideal pi pulse about y

    duration_s = t_max_ns * 1e-9
    n_samples = int(round(duration_s * fs)) + 1
    dt_ns = 1e9 / fs
    tlist = np.arange(n_samples) * dt_ns
    M = (n_samples - 1) // 2                                # echo half-time indices 0..M
    t_total = 2.0 * np.arange(M + 1) * dt_ns

    demo_half_idx = [
        int(np.clip(round(T / 2.0 / dt_ns), 1, M)) for T in demo_total_ns
    ]
    demo_sum = [np.zeros((2 * m + 1, 3)) for m in demo_half_idx]
    sum_bloch = np.zeros((M + 1, 3))

    options = {"progress_bar": "", "max_step": dt_ns, "atol": 1e-9, "rtol": 1e-7}
    seeds = np.random.SeedSequence(seed).spawn(n_realizations)

    t0 = time.time()
    for k, child in enumerate(seeds):
        if noise_rms > 0:
            _, noise_Hz = generate_pink_noise(
                duration=duration_s, fs=fs, amplitude=noise_rms,
                f_min=f_min, f_max=f_max,
                seed=int(child.generate_state(1)[0]),
                units=noise_units, dJ_dEps_MHz_per_mV=dJ_dEps_MHz_per_mV,
                output="Hz", n_fluctuators=n_fluctuators,
            )
            noise_rad_ns = two_pi * noise_Hz * 1e-9         # Hz -> rad/ns
        else:
            noise_rad_ns = np.zeros(n_samples)

        H = qt.QobjEvo([H0, [Hn, noise_rad_ns]], tlist=tlist, order=0)
        props = qt.propagator(H, tlist, options=options)
        U = np.array([p.full() for p in props])             # U(0 -> t_k), shape (n, 2, 2)

        # echo for every total time T_m = 2 m dt:  U(2m) U(m)^dag U_pi U(m) |phi>
        Um, U2m = U[: M + 1], U[0: 2 * M + 1: 2]
        v = np.einsum("mij,j->mi", Um, phi)
        v = v @ U_pi.T
        v = np.einsum("mji,mj->mi", Um.conj(), v)
        v = np.einsum("mij,mj->mi", U2m, v)
        sum_bloch += _bloch(v)

        # in-sequence curves (dephasing during the first half, refocusing in the second)
        for j, md in enumerate(demo_half_idx):
            first = np.einsum("mij,j->mi", U[: md + 1], phi)
            chi = U[md].conj().T @ (U_pi @ (U[md] @ phi))
            second = np.einsum("mij,j->mi", U[md: 2 * md + 1], chi)
            demo_sum[j] += np.vstack([_bloch(first), _bloch(second)[1:]])

        if verbose and (k + 1) % max(1, n_realizations // 5) == 0:
            print(f"  realization {k + 1}/{n_realizations}  ({time.time() - t0:.1f} s)")

    W = np.clip(np.linalg.norm(sum_bloch / n_realizations, axis=1), 0.0, 1.0)
    demo = [
        {
            "total_ns": 2.0 * md * dt_ns,
            "t_ns": np.arange(2 * md + 1) * dt_ns,
            "W": np.clip(np.linalg.norm(s / n_realizations, axis=1), 0.0, 1.0),
        }
        for md, s in zip(demo_half_idx, demo_sum)
    ]
    return {"t_total": t_total, "W": W, "demo": demo, "runtime_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hahn_echo_simulation(
    noise_rms: float = 10.0,
    noise_units: str = "uV",
    *,
    J0_MHz: float = J0_MHZ,
    dBz_MHz: float = DBZ_MHZ,
    dJ_dEps_MHz_per_mV: float = DJ_DEPS_MHZ_PER_MV,
    f_min: float = 1.0,
    f_max: float = 1e8,
    fs: float = 2e9,
    t_max_ns: float | None = None,
    n_realizations: int = 300,
    n_fluctuators: int | None = None,
    seed: int | None = 2026,
    w_fit_min: float = 0.15,
    verbose: bool = False,
) -> HahnEchoResult:
    """
    Simulate the Hahn echo and extract T2 (defined on the total time T = 2 tau).

    noise_rms          RMS of the detuning noise d_eps, in `noise_units`.
    f_min, f_max       Band [Hz] of the RTS corner frequencies (1/f range).
    fs                 Sampling rate of noise and propagators [Hz]
                       (keep >= ~10 x the 126 MHz splitting).
    t_max_ns           Longest total time; default = 3 x the analytic 1/f estimate.
    n_realizations     Noise realizations averaged (floor ~ 0.89/sqrt(N)).

    Returns a HahnEchoResult; `print(result)` shows T2 and all parameters.
    """
    if int(qt.__version__.split(".")[0]) < 5:
        raise RuntimeError("QuTiP >= 5 is required.")
    if noise_units not in _UNIT_TO_MV:
        raise ValueError("noise_units must be 'V', 'mV' or 'uV'.")
    if noise_rms <= 0:
        raise ValueError("noise_rms must be positive for a T2 fit.")

    t_echo_est = estimate_T2echo_1f_ns(
        noise_rms, noise_units, f_min, f_max, J0_MHz, dBz_MHz, dJ_dEps_MHz_per_mV
    )
    t_star_est = estimate_T2star_static_ns(
        noise_rms, noise_units, J0_MHz, dBz_MHz, dJ_dEps_MHz_per_mV
    )
    if t_max_ns is None:
        t_max_ns = 3.0 * t_echo_est

    demo_totals = [f * t_echo_est for f in (0.5, 1.0, 1.5)]
    demo_totals = [min(T, t_max_ns) for T in demo_totals]

    out = _run_ensemble(
        noise_rms, noise_units, J0_MHz, dBz_MHz, dJ_dEps_MHz_per_mV, f_min, f_max,
        fs, t_max_ns, n_realizations, n_fluctuators, seed, demo_totals, verbose,
    )
    fit = fit_echo(out["t_total"], out["W"], w_fit_min=w_fit_min)

    params = dict(
        noise_rms=noise_rms, noise_units=noise_units, J0_MHz=J0_MHz, dBz_MHz=dBz_MHz,
        dJ_dEps_MHz_per_mV=dJ_dEps_MHz_per_mV, f_min=f_min, f_max=f_max, fs=fs,
        t_max_ns=t_max_ns, n_realizations=n_realizations, seed=seed,
        T2echo_1f_estimate_ns=t_echo_est, T2star_static_estimate_ns=t_star_est,
        runtime_s=out["runtime_s"],
    )
    return HahnEchoResult(
        t_total_ns=out["t_total"], W=out["W"],
        T2_ns=fit["T2_ns"], alpha=fit["alpha"],
        T2_err_ns=fit["T2_err_ns"], alpha_err=fit["alpha_err"],
        T_1e_ns=fit["T_1e_ns"], fit_mask=fit["fit_mask"],
        demo=out["demo"], params=params,
    )


# ---------------------------------------------------------------------------
# Verification helper: noise-free refocusing
# ---------------------------------------------------------------------------

def noise_free_refocusing_error(
    J0_MHz: float = J0_MHZ, dBz_MHz: float = DBZ_MHZ, fs: float = 2e9, t_max_ns: float = 200.0
) -> float:
    """max |1 - W(T)| without noise: static J0 and dBz must be refocused exactly (~0)."""
    out = _run_ensemble(
        0.0, "uV", J0_MHz, dBz_MHz, DJ_DEPS_MHZ_PER_MV, 1.0, 1e8, fs,
        t_max_ns, 1, None, 0, [t_max_ns / 2.0], False,
    )
    return float(np.max(np.abs(1.0 - out["W"])))


# ---------------------------------------------------------------------------
# Plots (only used from __main__)
# ---------------------------------------------------------------------------

def plot_echo_formation(
    res: HahnEchoResult, save_path: str | None = "hahn_echo_formation.png", show: bool = True
):
    """Coherence during one echo sequence: dephasing, pi pulse, refocusing."""
    import matplotlib.pyplot as plt

    u = res.params["noise_units"].replace("uV", "µV")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(res.demo)))
    for d, c in zip(res.demo, colors):
        ax.plot(d["t_ns"], d["W"], color=c, lw=1.8, label=f"T = {d['total_ns']:.0f} ns")
        ax.axvline(d["total_ns"] / 2.0, color=c, ls=":", lw=1)
        ax.plot(d["total_ns"], d["W"][-1], "o", color=c)
    ax.set_xlabel("Time within the sequence (ns)")
    ax.set_ylabel("Coherence $|\\langle \\mathbf{r}\\rangle|$")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(d["total_ns"] for d in res.demo) * 1.02)
    ax.set_title(
        "Echo formation: dephasing, $\\pi$ pulse (dotted), refocusing\n"
        f"(noise RMS = {res.params['noise_rms']:g} {u}, N = {res.params['n_realizations']})"
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    return fig


def plot_echo_decay_fit(
    res: HahnEchoResult, save_path: str | None = "hahn_echo_decay_fit.png", show: bool = True
):
    """Echo decay W(T) with the stretched-exponential fit."""
    import matplotlib.pyplot as plt

    t, W, m = res.t_total_ns, res.W, res.fit_mask
    tt = np.linspace(0, t[-1], 600)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(t, W, ".", ms=3, color="tab:blue", alpha=0.7, label="Simulated echo $W(T)$")
    ax.plot(t[m], W[m], ".", ms=4, color="tab:orange", label="Points used in fit")
    ax.plot(
        tt, _stretched_exp(tt, res.T2_ns, res.alpha), "k-", lw=1.8,
        label=rf"Fit: $e^{{-(T/T_2)^{{\alpha}}}}$, $T_2$ = {res.T2_ns:.0f} ns, $\alpha$ = {res.alpha:.2f}",
    )
    ax.axhline(np.exp(-1), color="gray", ls=":", lw=1)
    ax.axvline(res.T2_ns, color="gray", ls=":", lw=1)
    ax.axvline(res.params["T2star_static_estimate_ns"], color="tab:red", ls="--", lw=1,
               label=r"quasi-static $T_2^*$ (Ramsey estimate)")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Total evolution time $T = 2\\tau$ (ns)")
    ax.set_ylabel("Echo coherence $W(T)$")
    ax.set_title("Hahn-echo decay and $T_2$ extraction")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Use case: run this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Set this to the charge-noise RMS you want to study (uV or mV).
    NOISE_RMS = 0.5
    NOISE_UNITS = "uV"

    print("Verification 1: noise-free refocusing (static J0 and dBz must cancel)")
    err = noise_free_refocusing_error()
    print(f"  max |1 - W(T)| = {err:.2e}   (expected ~ 0)\n")

    print("Verification 2: noisy Hahn echo and T2 extraction")
    result = hahn_echo_simulation(NOISE_RMS, NOISE_UNITS, verbose=True)
    print()
    print(result)

    plot_echo_formation(result)
    plot_echo_decay_fit(result)