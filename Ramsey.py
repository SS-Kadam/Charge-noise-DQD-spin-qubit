# Copyright (c) 2026 Shardul Kadam
# Licensed under the MIT License.
# See the LICENSE file in the project root for details.


"""
Ramsey.py
=============

Ramsey interferometry of an S-T0 spin qubit under RTS-sum (1/f) charge noise,
simulated with QuTiP (>= 5). Extracts T2* from the coherence envelope W(t).

Hamiltonian (hbar = 1, angular frequencies):

    H(t) = (1/2) * 2*pi * [ J0 + dJ/d_eps * d_eps(t) ] * sigma_x
         + (1/2) * 2*pi * dBz                          * sigma_z

with J0 = 119 MHz, dJ/d_eps = 160 MHz/mV, dBz = 42.1 MHz and d_eps(t) the
charge-noise trace from pink_noise.py (sum of random telegraph signals).

Units inside the solver: time in ns, angular frequency in rad/ns.
Public inputs/outputs use lab units (MHz, mV/uV, ns).

Ramsey protocol (ideal, instantaneous pi/2 pulses)
--------------------------------------------------
The noise-free Hamiltonian H0 has eigenstates |e0>, |e1> separated by
Omega = sqrt(J0^2 + dBz^2) = 126.2 MHz; its eigenaxis is tilted from x by
atan(dBz/J0) ~ 19.5 deg.

  1. First pi/2 pulse:  |phi> = (|e0> + |e1>)/sqrt(2)   (equator of H0)
  2. Free evolution for time t under the noisy H(t)
  3. Second pi/2 pulse + readout:  P(t) = |<phi|psi(t)>|^2     ("fringes")
         noise-free:  P(t) = (1 + cos(Omega t)) / 2

Coherence envelope (what is fitted):
  W(t) = 2 |< rho_01(t) >|  = sqrt(<X>^2 + <Y>^2),
  with rho in the H0 eigenbasis and < > the average over noise realizations.
  W = 1 without noise.  The fit model is  W(t) = exp(-(t/T2*)^alpha).

Plots/prints in the `__main__` block run only when this file is executed
directly (python ramsey_st0.py). Importing and calling
`ramsey_simulation(...)` never plots.
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


def estimate_T2star_static_ns(
    noise_rms: float,
    noise_units: str = "uV",
    J0_MHz: float = J0_MHZ,
    dBz_MHz: float = DBZ_MHZ,
    dJ_dEps_MHz_per_mV: float = DJ_DEPS_MHZ_PER_MV,
) -> float:
    """
    Quasi-static (Gaussian) estimate  T2* = sqrt(2) / (2 pi cos(theta) dJ/d_eps sigma_eps),
    cos(theta) = J0 / Omega being the projection of sigma_x noise on the eigenaxis.
    For 1/f noise the true T2* is a bit longer (part of the variance is fast).
    """
    sigma_mV = noise_rms * _UNIT_TO_MV[noise_units]
    cos_theta = J0_MHz / splitting_MHz(J0_MHz, dBz_MHz)
    sigma_long_MHz = cos_theta * dJ_dEps_MHz_per_mV * sigma_mV
    return float(np.sqrt(2.0) / (2.0 * np.pi * sigma_long_MHz * 1e-3))   # MHz -> cycles/ns


def _stretched_exp(t: NDArray, T2: float, alpha: float) -> NDArray:
    return np.exp(-(t / T2) ** alpha)


def fit_coherence(
    t_ns: NDArray[np.float64],
    W: NDArray[np.float64],
    w_fit_min: float = 0.15,
) -> dict:
    """
    Fit W(t) = exp(-(t/T2*)^alpha) on the points with W >= w_fit_min
    (the tail is dominated by the statistical floor ~ 0.89/sqrt(N)).
    Also returns the model-free 1/e time.
    """
    # model-free 1/e crossing (linear interpolation)
    below = np.where(W < np.exp(-1.0))[0]
    if below.size:
        k = below[0]
        t_1e = float(np.interp(np.exp(-1.0), [W[k], W[k - 1]], [t_ns[k], t_ns[k - 1]]))
    else:
        t_1e = float("nan")

    mask = W >= w_fit_min
    # use the contiguous region before W first drops below the threshold
    first_bad = np.where(~mask)[0]
    if first_bad.size:
        mask[first_bad[0]:] = False
    if mask.sum() < 5:
        raise RuntimeError("Too few points above w_fit_min; increase t_max or lower w_fit_min.")

    p0 = (t_1e if np.isfinite(t_1e) else float(t_ns[mask][-1]), 1.5)
    popt, pcov = curve_fit(
        _stretched_exp, t_ns[mask], W[mask], p0=p0,
        bounds=([1e-3, 0.3], [np.inf, 4.0]),
    )
    perr = np.sqrt(np.diag(pcov))
    return {
        "T2star_ns": float(popt[0]),
        "alpha": float(popt[1]),
        "T2star_err_ns": float(perr[0]),
        "alpha_err": float(perr[1]),
        "T_1e_ns": t_1e,
        "fit_mask": mask,
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RamseyResult:
    t_ns: NDArray[np.float64]
    fringe: NDArray[np.float64]        # ensemble-averaged P(t)
    W: NDArray[np.float64]             # coherence envelope W(t)
    T2star_ns: float
    alpha: float
    T2star_err_ns: float
    alpha_err: float
    T_1e_ns: float
    fit_mask: NDArray[np.bool_]
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def ramsey_simulation(
    noise_rms: float = 5.0,
    noise_units: str = "uV",
    *,
    J0_MHz: float = J0_MHZ,
    dBz_MHz: float = DBZ_MHZ,
    dJ_dEps_MHz_per_mV: float = DJ_DEPS_MHZ_PER_MV,
    f_min: float = 1.0,
    f_max: float = 1e8,
    fs: float = 2e9,
    t_max_ns: float | None = None,
    n_realizations: int = 500,
    n_fluctuators: int | None = None,
    seed: int | None = 2026,
    w_fit_min: float = 0.15,
    verbose: bool = False,
) -> RamseyResult:
    """
    Simulate Ramsey fringes and extract T2*.

    noise_rms          RMS of the detuning noise d_eps, in `noise_units`.
    f_min, f_max       Band [Hz] of the RTS corner frequencies (1/f range).
    fs                 Sampling rate of noise and output grid [Hz]
                       (>= ~10 x the 126 MHz splitting).
    t_max_ns           Simulated time; default = 4 x the quasi-static T2*.
    n_realizations     Noise realizations averaged (floor ~ 0.89/sqrt(N)).
    """
    if int(qt.__version__.split(".")[0]) < 5:
        raise RuntimeError("QuTiP >= 5 is required.")
    if noise_units not in _UNIT_TO_MV:
        raise ValueError("noise_units must be 'V', 'mV' or 'uV'.")

    # --- time grid ---------------------------------------------------------
    t_est = estimate_T2star_static_ns(
        noise_rms, noise_units, J0_MHz, dBz_MHz, dJ_dEps_MHz_per_mV
    ) if noise_rms > 0 else 1000.0
    if t_max_ns is None:
        t_max_ns = 4.0 * t_est
    duration_s = t_max_ns * 1e-9
    n_samples = int(round(duration_s * fs)) + 1
    tlist = np.arange(n_samples) / fs * 1e9                 # ns

    # --- operators and states (rad/ns) -------------------------------------
    two_pi = 2.0 * np.pi
    sx, sz = qt.sigmax(), qt.sigmaz()
    H0 = 0.5 * two_pi * J0_MHz * 1e-3 * sx + 0.5 * two_pi * dBz_MHz * 1e-3 * sz
    Hn = 0.5 * sx                                           # multiplies noise [rad/ns]

    _, evecs = H0.eigenstates()
    e0, e1 = evecs[0], evecs[1]
    phi = (e0 + e1).unit()                                  # state after first pi/2

    X = e0 * e1.dag() + e1 * e0.dag()
    Y = -1j * e0 * e1.dag() + 1j * e1 * e0.dag()
    P_phi = phi * phi.dag()                                 # second pi/2 + readout

    # --- ensemble ------------------------------------------------------------
    sum_X = np.zeros(n_samples)
    sum_Y = np.zeros(n_samples)
    sum_P = np.zeros(n_samples)

    seeds = np.random.SeedSequence(seed).spawn(n_realizations)
    dt_ns = 1e9 / fs
    options = {"progress_bar": "", "max_step": dt_ns, "atol": 1e-9, "rtol": 1e-7}

    t0 = time.time()
    for k, child in enumerate(seeds):
        if noise_rms > 0:
            _, noise_Hz = generate_pink_noise(
                duration=duration_s, fs=fs, amplitude=noise_rms,
                f_min=f_min, f_max=f_max,
                seed=int(child.generate_state(1)[0]),
                units=noise_units,
                dJ_dEps_MHz_per_mV=dJ_dEps_MHz_per_mV,
                output="Hz",
                n_fluctuators=n_fluctuators,
            )
            noise_rad_ns = two_pi * noise_Hz * 1e-9         # Hz -> rad/ns
        else:
            noise_rad_ns = np.zeros(n_samples)

        H = qt.QobjEvo([H0, [Hn, noise_rad_ns]], tlist=tlist, order=0)
        res = qt.sesolve(H, phi, tlist, e_ops=[X, Y, P_phi], options=options)

        sum_X += np.real(res.expect[0])
        sum_Y += np.real(res.expect[1])
        sum_P += np.real(res.expect[2])

        if verbose and (k + 1) % max(1, n_realizations // 5) == 0:
            print(f"  realization {k + 1}/{n_realizations}  ({time.time() - t0:.1f} s)")

    mean_X = sum_X / n_realizations
    mean_Y = sum_Y / n_realizations
    fringe = sum_P / n_realizations
    W = np.clip(np.hypot(mean_X, mean_Y), 0.0, 1.0)

    fit = fit_coherence(tlist, W, w_fit_min=w_fit_min)

    params = dict(
        noise_rms=noise_rms, noise_units=noise_units, J0_MHz=J0_MHz, dBz_MHz=dBz_MHz,
        dJ_dEps_MHz_per_mV=dJ_dEps_MHz_per_mV, f_min=f_min, f_max=f_max, fs=fs,
        t_max_ns=t_max_ns, n_realizations=n_realizations, seed=seed,
        T2star_static_estimate_ns=t_est, runtime_s=time.time() - t0,
    )
    return RamseyResult(
        t_ns=tlist, fringe=fringe, W=W,
        T2star_ns=fit["T2star_ns"], alpha=fit["alpha"],
        T2star_err_ns=fit["T2star_err_ns"], alpha_err=fit["alpha_err"],
        T_1e_ns=fit["T_1e_ns"], fit_mask=fit["fit_mask"], params=params,
    )


# ---------------------------------------------------------------------------
# Verification helper: noise-free fringe frequency
# ---------------------------------------------------------------------------

def noise_free_fringe_frequency_MHz(
    J0_MHz: float = J0_MHZ, dBz_MHz: float = DBZ_MHZ, fs: float = 2e9, t_max_ns: float = 400.0
) -> float:
    """Fringe frequency of the noise-free run (should equal sqrt(J0^2 + dBz^2))."""
    two_pi = 2.0 * np.pi
    H0 = (0.5 * two_pi * J0_MHz * 1e-3 * qt.sigmax()
          + 0.5 * two_pi * dBz_MHz * 1e-3 * qt.sigmaz())
    _, ev = H0.eigenstates()
    phi = (ev[0] + ev[1]).unit()
    tlist = np.arange(int(round(t_max_ns * 1e-9 * fs)) + 1) / fs * 1e9      # ns
    res = qt.sesolve(H0, phi, tlist, e_ops=[phi * phi.dag()],
                     options={"progress_bar": ""})
    p = np.real(res.expect[0])
    p = (p - p.mean()) * np.hanning(p.size)
    n_pad = 2 ** 17
    spec = np.abs(np.fft.rfft(p, n=n_pad))
    freqs_cycles_per_ns = np.fft.rfftfreq(n_pad, d=tlist[1] - tlist[0])
    return float(freqs_cycles_per_ns[np.argmax(spec)] * 1e3)                # -> MHz


# ---------------------------------------------------------------------------
# Plots (only used from __main__)
# ---------------------------------------------------------------------------

def plot_fringes(res: RamseyResult, save_path: str | None = "ramsey_fringes.png", show: bool = True):
    import matplotlib.pyplot as plt

    t, P, W = res.t_ns, res.fringe, res.W
    u = res.params["noise_units"].replace("uV", "µV")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.2))

    ax1.plot(t, P, lw=0.6, color="tab:blue", label="Ramsey fringes  $P(t)$")
    ax1.plot(t, 0.5 * (1 + W), "k--", lw=1.2, label=r"envelope $\frac{1}{2}(1\pm W)$")
    ax1.plot(t, 0.5 * (1 - W), "k--", lw=1.2)
    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("Free-evolution time (ns)")
    ax1.set_ylabel("Return probability")
    ax1.set_title(
        f"Ramsey fringes, S-T0 qubit  (noise RMS = {res.params['noise_rms']:g} {u}, "
        f"N = {res.params['n_realizations']})"
    )
    ax1.legend(loc="upper right")

    zoom = t <= min(t[-1], 40.0)
    ax2.plot(t[zoom], P[zoom], "o-", ms=3, lw=1, color="tab:blue")
    ax2.set_xlabel("Free-evolution time (ns)")
    ax2.set_ylabel("Return probability")
    ax2.set_title(
        "Zoom: first 40 ns "
        f"(fringe period = {1e3 / splitting_MHz(res.params['J0_MHz'], res.params['dBz_MHz']):.2f} ns)"
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    return fig


def plot_coherence_fit(res: RamseyResult, save_path: str | None = "ramsey_W_fit.png", show: bool = True):
    import matplotlib.pyplot as plt

    t, W, m = res.t_ns, res.W, res.fit_mask
    tt = np.linspace(0, t[-1], 600)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(t, W, ".", ms=3, color="tab:blue", alpha=0.7, label="Simulated $W(t)$")
    ax.plot(t[m], W[m], ".", ms=4, color="tab:orange", label="Points used in fit")
    ax.plot(
        tt, _stretched_exp(tt, res.T2star_ns, res.alpha), "k-", lw=1.8,
        label=rf"Fit: $e^{{-(t/T_2^*)^{{\alpha}}}}$, $T_2^*$ = {res.T2star_ns:.0f} ns, $\alpha$ = {res.alpha:.2f}",
    )
    ax.axhline(np.exp(-1), color="gray", ls=":", lw=1)
    ax.axvline(res.T2star_ns, color="gray", ls=":", lw=1)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Free-evolution time (ns)")
    ax.set_ylabel("Coherence $W(t)$")
    ax.set_title("Coherence decay and $T_2^*$ extraction")
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
    # Set this to the charge-noise RMS reported in the paper (uV or mV).
    NOISE_RMS = 5.0
    NOISE_UNITS = "uV"

    print("Verification 1: noise-free fringe frequency")
    f_sim = noise_free_fringe_frequency_MHz()
    print(f"  simulated = {f_sim:.1f} MHz   expected sqrt(J0^2+dBz^2) = {splitting_MHz():.1f} MHz\n")

    print("Verification 2: noisy Ramsey and T2* extraction")
    result = ramsey_simulation(NOISE_RMS, NOISE_UNITS, verbose=True)

    u = NOISE_UNITS.replace("uV", "µV")
    p = result.params
    print("\n========== Ramsey simulation (S-T0 qubit) ==========")
    print(f"J0 = {p['J0_MHz']:g} MHz, dBz = {p['dBz_MHz']:g} MHz, dJ/d_eps = {p['dJ_dEps_MHz_per_mV']:g} MHz/mV")
    print(f"Noise RMS            = {p['noise_rms']:g} {u}   (J-noise {p['noise_rms'] * _UNIT_TO_MV[NOISE_UNITS] * p['dJ_dEps_MHz_per_mV']:.3g} MHz)")
    print(f"Noise band           = {p['f_min']:g} - {p['f_max']:.3g} Hz,  realizations = {p['n_realizations']}")
    print(f"T2* (stretched fit)  = {result.T2star_ns:.1f} +/- {result.T2star_err_ns:.1f} ns")
    print(f"Stretch exponent     = {result.alpha:.2f} +/- {result.alpha_err:.2f}")
    print(f"T2* (1/e crossing)   = {result.T_1e_ns:.1f} ns")
    print(f"Quasi-static estimate= {p['T2star_static_estimate_ns']:.1f} ns  (Gaussian limit; 1/f gives slightly longer)")
    print(f"Runtime              = {p['runtime_s']:.1f} s")
    print("=====================================================")

    plot_fringes(result)
    plot_coherence_fit(result)