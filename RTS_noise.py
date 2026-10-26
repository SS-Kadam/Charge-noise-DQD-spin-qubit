# Copyright (c) 2026 Shardul Kadam
# Licensed under the MIT License.
# See the LICENSE file in the project root for details.

"""
RTS_noise.py
=============

Charge (detuning) noise built as a sum of random telegraph signals (RTS),
i.e. the McWhorter picture of 1/f noise: many two-level fluctuators (TLFs)
whose switching rates are spread log-uniformly over a band produce a 1/f
spectrum in that band.

Physical conventions
--------------------
* Each fluctuator i is a symmetric telegraph signal x_i(t) = +/-1 that flips
  at rate Gamma_i (per state). Its correlation function is exp(-2*Gamma_i*|t|),
  i.e. a Lorentzian PSD with corner frequency  f_c,i = Gamma_i / pi.
* The detuning noise is   d_eps(t) = A * sum_i w_i x_i(t),   sum_i w_i^2 = 1,
  so A is the RMS of d_eps (in the units you pass in: V, mV or uV).
* With corner frequencies spread log-uniformly in [f_min, f_max]
  the one-sided PSD in that band is
        S_eps(f) = A^2 / ( f * ln(f_max / f_min) ).
* The qubit sees the noise through the exchange coupling:
        J(t) = J0 + (dJ/d_eps) * d_eps(t)
  so the simulation noise is  noise_Hz = dJ_dEps[Hz/mV] * d_eps[mV].
  It is a shift of J expressed in Hz (cycles/s), NOT angular frequency.
  In the S-T0 Hamiltonian it multiplies sigma_x:

        H = (hbar/2) * [ 2*pi*(J0_Hz + noise_Hz(t)) ] sigma_x
          + (hbar/2) * [ 2*pi*dBz_Hz ] sigma_z

  Use output="angular" (rad/s) if you want the 2*pi already applied.

The process is simulated *exactly* at the grid points (a two-state Markov
chain with flip probability p = (1 - exp(-2*pi*f_c*dt)) / 2 per step) and
each fluctuator starts from its stationary distribution, so no burn-in,
no long record and no window trick is needed. Fluctuators slower than
1/duration simply appear as random static offsets from shot to shot, which
is the correct physics for T2*.

The plot at the bottom only runs when this file is executed directly
(`python pink_noise.py`); importing or calling the functions never plots.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Defaults (from the S-T0 Hamiltonian discussed for the project)
# ---------------------------------------------------------------------------

DEFAULT_DJ_DEPS_MHZ_PER_MV = 160.0   # dJ/d_eps  [MHz/mV]

_TO_MV = {"V": 1e3, "mV": 1.0, "uV": 1e-3, "µV": 1e-3}
_LABEL = {"V": "V", "mV": "mV", "uV": "µV", "µV": "µV"}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NoiseRealization:
    """One noise realization plus the fluctuator set that produced it."""

    time_s: NDArray[np.float64]            # time axis [s]
    eps: NDArray[np.float64]               # detuning noise, in `units`
    units: str                             # "V", "mV" or "uV"
    amplitude: float                       # target RMS of eps, in `units`
    corner_freqs_hz: NDArray[np.float64]   # f_c,i of every fluctuator [Hz]
    weights: NDArray[np.float64]           # w_i, with sum(w_i^2) = 1
    dJ_dEps_MHz_per_mV: float

    @property
    def eps_mV(self) -> NDArray[np.float64]:
        return self.eps * _TO_MV[self.units]

    @property
    def noise_Hz(self) -> NDArray[np.float64]:
        """Shift of J in Hz (cycles/s)."""
        return self.eps_mV * self.dJ_dEps_MHz_per_mV * 1e6

    @property
    def noise_rad_s(self) -> NDArray[np.float64]:
        """Shift of J in rad/s (2*pi * Hz)."""
        return 2.0 * np.pi * self.noise_Hz


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _corner_frequencies(
    f_min: float,
    f_max: float,
    n: int,
    sampling: str,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Corner frequencies spread log-uniformly in [f_min, f_max]."""
    if sampling == "random":
        u = rng.random(n)
    elif sampling == "stratified":
        u = (np.arange(n) + rng.random(n)) / n      # one draw per log-bin
    elif sampling == "regular":
        u = (np.arange(n) + 0.5) / n                # deterministic
    else:
        raise ValueError("corner_sampling must be 'random', 'stratified' or 'regular'.")
    return np.exp(np.log(f_min) + u * (np.log(f_max) - np.log(f_min)))


def _rts_trace(n: int, p_flip: float, rng: np.random.Generator) -> NDArray[np.int64]:
    """
    +/-1 samples of a symmetric telegraph signal at n grid points.

    p_flip is the probability that the state differs between consecutive
    grid points; the initial state is drawn from the stationary distribution.
    """
    x0 = 1 - 2 * int(rng.integers(0, 2))
    if n <= 1:
        return np.full(n, x0, dtype=np.int64)

    steps = n - 1
    flips = np.zeros(n, dtype=np.int64)

    if p_flip >= 0.05:
        flips[1:] = rng.random(steps) < p_flip
    else:
        # Slow fluctuator: draw geometric gaps between flips instead of
        # one random number per sample (exactly equivalent).
        expected = steps * p_flip
        m = int(expected + 6.0 * np.sqrt(expected) + 16)
        pos = np.cumsum(rng.geometric(p_flip, size=m))
        while pos[-1] <= steps:
            more = np.cumsum(rng.geometric(p_flip, size=m)) + pos[-1]
            pos = np.concatenate([pos, more])
        flips[pos[pos <= steps]] = 1

    parity = np.cumsum(flips) & 1
    return x0 * (1 - 2 * parity)


def analytic_psd(
    freqs_hz: NDArray[np.float64],
    corner_freqs_hz: NDArray[np.float64],
    weights: NDArray[np.float64],
    amplitude: float,
) -> NDArray[np.float64]:
    """
    One-sided PSD of the fluctuator sum, in amplitude-units^2 / Hz:
        S(f) = A^2 * sum_i w_i^2 * 4*tau_i / (1 + (2*pi*f*tau_i)^2),
        tau_i = 1 / (2*pi*f_c,i).
    """
    f = np.asarray(freqs_hz, dtype=float)[:, None]
    tau = 1.0 / (2.0 * np.pi * np.asarray(corner_freqs_hz, dtype=float))[None, :]
    w2 = np.asarray(weights, dtype=float)[None, :] ** 2
    return amplitude**2 * np.sum(w2 * 4.0 * tau / (1.0 + (2.0 * np.pi * f * tau) ** 2), axis=1)


def ideal_pink_psd(
    freqs_hz: NDArray[np.float64], amplitude: float, f_min: float, f_max: float
) -> NDArray[np.float64]:
    """Ideal in-band result  A^2 / (f ln(f_max/f_min))  [amplitude-units^2/Hz]."""
    return amplitude**2 / (np.asarray(freqs_hz, dtype=float) * np.log(f_max / f_min))


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_rts_noise(
    duration: float,
    fs: float,
    amplitude: float,
    f_min: float,
    f_max: float,
    *,
    units: str = "uV",
    dJ_dEps_MHz_per_mV: float = DEFAULT_DJ_DEPS_MHZ_PER_MV,
    n_fluctuators: int | None = None,
    corner_freqs: NDArray[np.float64] | list[float] | None = None,
    corner_sampling: str = "stratified",
    coupling_spread: float = 0.0,
    seed: int | None = None,
    verbose: bool = False,
) -> NoiseRealization:
    """
    Generate one realization of RTS-sum detuning noise.

    Parameters
    ----------
    duration, fs      Record length [s] and sampling rate [Hz]; the record has
                      int(round(duration*fs)) + 1 samples.
    amplitude         RMS of the detuning noise, in `units`.
    f_min, f_max      Band [Hz] over which fluctuator corner frequencies are
                      spread log-uniformly (this sets the 1/f range).
    units             "V", "mV" or "uV": units of `amplitude` (and of the
                      printed summary).
    dJ_dEps_MHz_per_mV  Coupling used to convert detuning noise to J noise.
    n_fluctuators     Number of RTS. Default: 6 per decade of band (min 8).
    corner_freqs      Explicit corner frequencies [Hz]; overrides
                      f_min/f_max/n_fluctuators. Use a single value to study
                      one TLF (e.g. scan its rate against the coupling).
    corner_sampling   "stratified" (default), "random" or "regular".
    coupling_spread   Log-normal spread of the individual fluctuator weights
                      (0 = all equal). Total RMS stays equal to `amplitude`.
    seed              Seed for reproducibility.
    verbose           Print a summary in `units`.
    """
    if units not in _TO_MV:
        raise ValueError("units must be 'V', 'mV' or 'uV'.")
    if duration <= 0 or fs <= 0 or amplitude < 0:
        raise ValueError("duration and fs must be positive, amplitude non-negative.")

    rng = np.random.default_rng(seed)

    # --- fluctuator set -----------------------------------------------------
    if corner_freqs is not None:
        fc = np.atleast_1d(np.asarray(corner_freqs, dtype=float))
        if np.any(fc <= 0):
            raise ValueError("corner_freqs must be positive.")
    else:
        if not (0 < f_min < f_max):
            raise ValueError("Need 0 < f_min < f_max.")
        n = n_fluctuators
        if n is None:
            n = max(8, int(np.ceil(6.0 * np.log10(f_max / f_min))))
        fc = _corner_frequencies(f_min, f_max, int(n), corner_sampling, rng)

    n_fl = fc.size

    if np.max(fc) > fs / 2:
        warnings.warn(
            f"Fastest corner frequency ({np.max(fc):.3e} Hz) exceeds Nyquist "
            f"({fs / 2:.3e} Hz); increase fs or lower f_max.",
            stacklevel=2,
        )

    w = np.exp(coupling_spread * rng.standard_normal(n_fl))
    w /= np.sqrt(np.sum(w**2))                      # sum(w^2) = 1

    # --- time grid and noise ------------------------------------------------
    dt = 1.0 / fs
    n_samples = int(round(duration * fs)) + 1
    time_s = np.arange(n_samples) * dt

    eps = np.zeros(n_samples, dtype=np.float64)
    for f_c, w_i in zip(fc, w):
        p_flip = -0.5 * np.expm1(-2.0 * np.pi * f_c * dt)   # (1 - e^{-2 pi f_c dt}) / 2
        eps += w_i * _rts_trace(n_samples, p_flip, rng)
    eps *= amplitude

    result = NoiseRealization(
        time_s=time_s,
        eps=eps,
        units=units,
        amplitude=float(amplitude),
        corner_freqs_hz=fc,
        weights=w,
        dJ_dEps_MHz_per_mV=float(dJ_dEps_MHz_per_mV),
    )

    if verbose:
        _print_summary(result, duration, fs, corner_sampling if corner_freqs is None else "user")

    return result


def generate_pink_noise(
    duration: float,
    fs: float,
    amplitude: float,
    f_min: float,
    f_max: float,
    seed: int | None = None,
    *,
    units: str = "uV",
    dJ_dEps_MHz_per_mV: float = DEFAULT_DJ_DEPS_MHZ_PER_MV,
    output: str = "Hz",
    n_fluctuators: int | None = None,
    corner_freqs: NDArray[np.float64] | list[float] | None = None,
    corner_sampling: str = "stratified",
    coupling_spread: float = 0.0,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Drop-in generator returning (time_s, noise).

    `amplitude` is the RMS detuning noise in `units` ("V", "mV", "uV").
    `output` selects what `noise` contains:
        "Hz"       shift of J in Hz           (default; multiply by 2*pi in H)
        "angular"  shift of J in rad/s        (2*pi already applied)
        "voltage"  the detuning noise itself, in `units`
    """
    rec = generate_rts_noise(
        duration, fs, amplitude, f_min, f_max,
        units=units,
        dJ_dEps_MHz_per_mV=dJ_dEps_MHz_per_mV,
        n_fluctuators=n_fluctuators,
        corner_freqs=corner_freqs,
        corner_sampling=corner_sampling,
        coupling_spread=coupling_spread,
        seed=seed,
        verbose=verbose,
    )

    if output == "Hz":
        noise = rec.noise_Hz
    elif output == "angular":
        noise = rec.noise_rad_s
    elif output == "voltage":
        noise = rec.eps
    else:
        raise ValueError("output must be 'Hz', 'angular' or 'voltage'.")

    return rec.time_s, noise


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_summary(rec: NoiseRealization, duration: float, fs: float, sampling: str) -> None:
    u = _LABEL[rec.units]
    fc = rec.corner_freqs_hz
    j_rms_mhz = rec.amplitude * _TO_MV[rec.units] * rec.dJ_dEps_MHz_per_mV
    print("========== RTS charge-noise (McWhorter) ==========")
    print(f"Fluctuators          = {fc.size}  ({sampling} log-spaced corners)")
    print(f"Corner band          = {fc.min():.3e} - {fc.max():.3e} Hz")
    print(f"Target RMS  d_eps    = {rec.amplitude:.4g} {u}")
    print(f"Sample RMS  d_eps    = {np.sqrt(np.mean(rec.eps**2)):.4g} {u}   (this realization, about zero)")
    print(f"J-noise RMS          = {j_rms_mhz:.4g} MHz   (dJ/d_eps = {rec.dJ_dEps_MHz_per_mV:g} MHz/mV)")
    print(f"Record               = {duration:.3e} s @ {fs:.3e} Hz  ({rec.eps.size} samples)")
    slow = int(np.sum(fc < 1.0 / duration))
    if slow:
        print(f"Note: {slow} fluctuator(s) slower than 1/duration act as static offsets per shot.")
    print("==================================================")


# ---------------------------------------------------------------------------
# PSD visualisation (only runs when the file is executed directly)
# ---------------------------------------------------------------------------

def plot_psd(
    *,
    duration: float = 5e-4,
    fs: float = 1e9,
    amplitude: float = 10.0,
    f_min: float = 1e5,
    f_max: float = 1e8,
    units: str = "uV",
    n_realizations: int = 8,
    n_fluctuators: int | None = None,
    corner_sampling: str = "stratified",
    seed: int | None = 1234,
    nperseg: int = 2**16,
    save_path: str | None = None,
) -> None:
    """Welch PSD of the generated charge noise vs. analytic and ideal 1/f."""
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    u = _LABEL[units]
    psd_sum = None
    analytic_sum = None
    f_ref = None

    for k in range(n_realizations):
        rec = generate_rts_noise(
            duration, fs, amplitude, f_min, f_max,
            units=units,
            n_fluctuators=n_fluctuators,
            corner_sampling=corner_sampling,
            seed=None if seed is None else seed + k,
            verbose=(k == 0),
        )
        f, pxx = welch(rec.eps, fs=fs, nperseg=min(nperseg, rec.eps.size))
        f, pxx = f[1:], pxx[1:]                         # drop DC bin
        a = analytic_psd(f, rec.corner_freqs_hz, rec.weights, amplitude)
        psd_sum = pxx if psd_sum is None else psd_sum + pxx
        analytic_sum = a if analytic_sum is None else analytic_sum + a
        f_ref = f

    psd = psd_sum / n_realizations
    analytic = analytic_sum / n_realizations
    ideal = ideal_pink_psd(f_ref, amplitude, f_min, f_max)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.loglog(f_ref, psd, lw=1.2, label=f"Simulated (Welch, {n_realizations} realizations)")
    ax.loglog(f_ref, analytic, "--", lw=1.5, label="Sum of Lorentzians (analytic)")
    band = (f_ref >= f_min) & (f_ref <= f_max)
    ax.loglog(f_ref[band], ideal[band], ":", lw=2, color="k",
              label=r"Ideal $A^2/(f\,\ln(f_{max}/f_{min}))$")
    ax.axvline(f_min, color="gray", lw=0.8, ls="-.")
    ax.axvline(f_max, color="gray", lw=0.8, ls="-.")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Detuning-noise PSD ({u}$^2$/Hz)")
    ax.set_title(f"Charge noise from RTS sum  (RMS = {amplitude:g} {u})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    plot_psd()