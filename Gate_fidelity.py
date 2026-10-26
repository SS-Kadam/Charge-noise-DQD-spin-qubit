# Copyright (c) 2026 Shardul Kadam
# Licensed under the MIT License.
# See the LICENSE file in the project root for details.


"""
Gate_fidelity.py
====================

Direct average-gate-fidelity simulation for an S-T0 spin qubit under RTS-sum
(1/f) charge noise, using QuTiP (>= 5). Companion to ramsey_st0.py and
hahn_echo_st0.py (same Hamiltonian, units and noise generator).

    H(t) = (1/2) 2*pi [ J0 + dJ/d_eps * d_eps(t) ] sigma_x + (1/2) 2*pi dBz sigma_z

Gate definition
---------------
The target gate is the noise-free evolution under H0 for the gate time t_gate:

    U_target = exp(-i H0 t_gate)

(an exchange pulse of duration t_gate with the gradient always on). For each
noise realization the noisy propagator U(t_gate) is computed with QuTiP, and

    F_pro  = | Tr(U_target^dagger U) |^2 / d^2        (process fidelity, d = 2)
    F_avg  = (d * F_pro + 1) / (d + 1)                (average gate fidelity)

Both are averaged over noise realizations. (F_pro is linear in the channel,
so averaging per-realization values is exact for classical noise.)

Note on t_gate: 4.2 ns = 1/(2 J0) is a pi rotation about x for J0 alone. With
the gradient included the rotation angle is Omega*t_gate = 191 deg about the
tilted eigenaxis; an exact pi rotation takes 1/(2 Omega) = 3.96 ns. The
fidelity is defined relative to whatever H0 does in t_gate, so both are valid:
set t_gate_ns to the value you use for the R = T2*/t_gate metric.

Comparison with the ratio estimate: for quasi-static noise
    1 - F_avg ~ (sigma_par * t_gate)^2 / 6 = t_gate^2 / (3 T2*_static^2),
where T2*_static = sqrt(2)/sigma_par. The estimate 1/(3 R^2) with the *fitted*
Ramsey T2* is ~20 % lower, because part of the 1/f variance is faster than
1/T2* yet still acts as an error during the (very short) gate.

Plots and prints in the `__main__` block run only when this file is executed
directly. Importing and calling `gate_fidelity_simulation(...)` is silent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import qutip as qt
from numpy.typing import NDArray

from RTS_noise import generate_pink_noise

# ---------------------------------------------------------------------------
# Device parameters (from the paper's S-T0 Hamiltonian)
# ---------------------------------------------------------------------------

J0_MHZ = 119.0
DBZ_MHZ = 42.1
DJ_DEPS_MHZ_PER_MV = 160.0
T_GATE_NS = 4.2

_UNIT_TO_MV = {"V": 1e3, "mV": 1.0, "uV": 1e-3, "µV": 1e-3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def splitting_MHz(J0_MHz: float = J0_MHZ, dBz_MHz: float = DBZ_MHZ) -> float:
    return float(np.hypot(J0_MHz, dBz_MHz))


def _sigma_long_rad_per_ns(
    noise_rms: float, noise_units: str, J0_MHz: float, dBz_MHz: float,
    dJ_dEps_MHz_per_mV: float,
) -> float:
    sigma_mV = noise_rms * _UNIT_TO_MV[noise_units]
    cos_theta = J0_MHz / splitting_MHz(J0_MHz, dBz_MHz)
    return 2.0 * np.pi * cos_theta * dJ_dEps_MHz_per_mV * sigma_mV * 1e-3


def estimate_infidelity_static(
    noise_rms: float, noise_units: str = "uV", t_gate_ns: float = T_GATE_NS,
    J0_MHz: float = J0_MHZ, dBz_MHz: float = DBZ_MHZ,
    dJ_dEps_MHz_per_mV: float = DJ_DEPS_MHZ_PER_MV,
) -> float:
    """Quasi-static estimate  1 - F ~ (sigma_par t_gate)^2 / 6  (= 1/(3 R^2) with static T2*)."""
    s = _sigma_long_rad_per_ns(noise_rms, noise_units, J0_MHz, dBz_MHz, dJ_dEps_MHz_per_mV)
    return float((s * t_gate_ns) ** 2 / 6.0)


def _step_unitary(wx: float, wz: float, dt: float) -> NDArray[np.complex128]:
    w = np.hypot(wx, wz)
    c, s = np.cos(w * dt / 2.0), np.sin(w * dt / 2.0) / w
    return np.array([[c - 1j * s * wz, -1j * s * wx], [-1j * s * wx, c + 1j * s * wz]])


def exact_propagator(
    noise_rad_ns: NDArray[np.float64], J0_MHz: float, dBz_MHz: float, dt_ns: float
) -> NDArray[np.complex128]:
    """Reference: product of exact step unitaries for piecewise-constant noise."""
    two_pi = 2.0 * np.pi
    wz = two_pi * dBz_MHz * 1e-3
    U = np.eye(2, dtype=complex)
    for k in range(len(noise_rad_ns) - 1):
        U = _step_unitary(two_pi * J0_MHz * 1e-3 + noise_rad_ns[k], wz, dt_ns) @ U
    return U


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GateFidelityResult:
    F_avg: float                      # average gate fidelity
    infidelity: float                 # 1 - F_avg
    infidelity_sem: float             # standard error of the mean over realizations
    F_pro: float                      # process (entanglement) fidelity
    infidelity_samples: NDArray[np.float64] = field(repr=False, default_factory=lambda: np.array([]))
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        p = self.params
        u = p["noise_units"].replace("uV", "µV")
        return "\n".join([
            "========== Gate-fidelity simulation (S-T0 qubit) ==========",
            f"J0 = {p['J0_MHz']:g} MHz, dBz = {p['dBz_MHz']:g} MHz, dJ/d_eps = {p['dJ_dEps_MHz_per_mV']:g} MHz/mV",
            f"Gate time            = {p['t_gate_ns']:g} ns   (rotation {p['rotation_deg']:.1f} deg about the eigenaxis)",
            f"Noise RMS            = {p['noise_rms']:g} {u},  band {p['f_min']:g} - {p['f_max']:.3g} Hz",
            f"Realizations         = {p['n_realizations']},  steps per gate = {p['n_steps']},  seed = {p['seed']}",
            f"Average gate fidelity= {self.F_avg:.6f}   (F_pro = {self.F_pro:.6f})",
            f"Infidelity 1 - F     = {self.infidelity:.3e} +/- {self.infidelity_sem:.1e}",
            f"Static estimate      = {p['infidelity_static_estimate']:.3e}   (t_gate^2 / (3 T2*_static^2))",
            f"Runtime              = {p['runtime_s']:.1f} s",
            "===========================================================",
        ])

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def gate_fidelity_simulation(
    noise_rms: float = 10.0,
    noise_units: str = "uV",
    *,
    t_gate_ns: float = T_GATE_NS,
    J0_MHz: float = J0_MHZ,
    dBz_MHz: float = DBZ_MHZ,
    dJ_dEps_MHz_per_mV: float = DJ_DEPS_MHZ_PER_MV,
    f_min: float = 1.0,
    f_max: float = 1e8,
    n_realizations: int = 2000,
    n_steps: int = 42,
    n_fluctuators: int | None = None,
    seed: int | None = 2026,
) -> GateFidelityResult:
    """
    Average gate fidelity of the noisy gate exp(-i H0 t_gate).

    noise_rms        RMS of the detuning noise d_eps, in `noise_units`.
    t_gate_ns        Gate duration (use the same value as in R = T2*/t_gate).
    n_steps          Noise samples per gate (dt = t_gate/n_steps; 0.1 ns by default).
    n_realizations   Noise realizations. Relative standard error of 1 - F is
                     about sqrt(2/N) (~3 % for 2000).
    """
    if int(qt.__version__.split(".")[0]) < 5:
        raise RuntimeError("QuTiP >= 5 is required.")
    if noise_units not in _UNIT_TO_MV:
        raise ValueError("noise_units must be 'V', 'mV' or 'uV'.")

    two_pi = 2.0 * np.pi
    sx, sz = qt.sigmax(), qt.sigmaz()
    H0 = 0.5 * two_pi * J0_MHz * 1e-3 * sx + 0.5 * two_pi * dBz_MHz * 1e-3 * sz   # rad/ns
    Hn = 0.5 * sx
    U_target = (-1j * H0 * t_gate_ns).expm()
    U_target_dag = U_target.dag().full()

    dt_ns = t_gate_ns / n_steps
    fs = 1e9 / dt_ns
    tlist = np.linspace(0.0, t_gate_ns, n_steps + 1)
    options = {"progress_bar": "", "max_step": dt_ns, "atol": 1e-12, "rtol": 1e-10}

    seeds = np.random.SeedSequence(seed).spawn(n_realizations)
    F_pro = np.empty(n_realizations)

    t0 = time.time()
    for k, child in enumerate(seeds):
        _, noise_Hz = generate_pink_noise(
            duration=t_gate_ns * 1e-9, fs=fs, amplitude=noise_rms,
            f_min=f_min, f_max=f_max, seed=int(child.generate_state(1)[0]),
            units=noise_units, dJ_dEps_MHz_per_mV=dJ_dEps_MHz_per_mV,
            output="Hz", n_fluctuators=n_fluctuators,
        )
        noise_rad_ns = two_pi * noise_Hz * 1e-9
        if noise_rad_ns.size != tlist.size:
            raise RuntimeError("Noise grid does not match the gate time grid.")

        H = qt.QobjEvo([H0, [Hn, noise_rad_ns]], tlist=tlist, order=0)
        U = qt.propagator(H, t_gate_ns, options=options).full()
        F_pro[k] = abs(np.trace(U_target_dag @ U)) ** 2 / 4.0

    infid = (2.0 / 3.0) * (1.0 - F_pro)                    # 1 - F_avg per realization
    mean_infid = float(infid.mean())
    sem = float(infid.std(ddof=1) / np.sqrt(n_realizations))

    omega_rad_ns = np.hypot(two_pi * J0_MHz * 1e-3, two_pi * dBz_MHz * 1e-3)
    params = dict(
        noise_rms=noise_rms, noise_units=noise_units, t_gate_ns=t_gate_ns,
        J0_MHz=J0_MHz, dBz_MHz=dBz_MHz, dJ_dEps_MHz_per_mV=dJ_dEps_MHz_per_mV,
        f_min=f_min, f_max=f_max, n_realizations=n_realizations, n_steps=n_steps,
        seed=seed, rotation_deg=float(np.degrees(omega_rad_ns * t_gate_ns)),
        infidelity_static_estimate=estimate_infidelity_static(
            noise_rms, noise_units, t_gate_ns, J0_MHz, dBz_MHz, dJ_dEps_MHz_per_mV),
        runtime_s=time.time() - t0,
    )
    return GateFidelityResult(
        F_avg=1.0 - mean_infid, infidelity=mean_infid, infidelity_sem=sem,
        F_pro=float(F_pro.mean()), infidelity_samples=infid, params=params,
    )


def fidelity_sweep(noise_values: list[float], **kwargs) -> list[GateFidelityResult]:
    """Run gate_fidelity_simulation over several noise amplitudes (same units/settings)."""
    return [gate_fidelity_simulation(v, **kwargs) for v in noise_values]


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def noise_free_fidelity(t_gate_ns: float = T_GATE_NS) -> float:
    """Fidelity without noise (must be 1 up to solver accuracy)."""
    two_pi = 2.0 * np.pi
    H0 = 0.5 * two_pi * J0_MHZ * 1e-3 * qt.sigmax() + 0.5 * two_pi * DBZ_MHZ * 1e-3 * qt.sigmaz()
    U = qt.propagator(H0, t_gate_ns, options={"progress_bar": "", "atol": 1e-12, "rtol": 1e-10})
    Ut = (-1j * H0 * t_gate_ns).expm()
    return float(abs((Ut.dag() * U).tr()) ** 2 / 4.0)


def propagator_check(
    noise_rms: float = 20.0, noise_units: str = "uV", t_gate_ns: float = T_GATE_NS,
    n_steps: int = 42, seed: int = 7,
) -> float:
    """max |U_qutip - U_exact| for one noise realization (should be ~1e-6 or smaller)."""
    two_pi = 2.0 * np.pi
    dt_ns = t_gate_ns / n_steps
    _, noise_Hz = generate_pink_noise(
        duration=t_gate_ns * 1e-9, fs=1e9 / dt_ns, amplitude=noise_rms,
        f_min=1.0, f_max=1e8, seed=seed, units=noise_units, output="Hz",
    )
    noise_rad_ns = two_pi * noise_Hz * 1e-9
    tlist = np.linspace(0.0, t_gate_ns, n_steps + 1)
    H0 = 0.5 * two_pi * J0_MHZ * 1e-3 * qt.sigmax() + 0.5 * two_pi * DBZ_MHZ * 1e-3 * qt.sigmaz()
    H = qt.QobjEvo([H0, [0.5 * qt.sigmax(), noise_rad_ns]], tlist=tlist, order=0)
    U_q = qt.propagator(H, t_gate_ns, options={"progress_bar": "", "max_step": dt_ns,
                                               "atol": 1e-12, "rtol": 1e-10}).full()
    U_e = exact_propagator(noise_rad_ns, J0_MHZ, DBZ_MHZ, dt_ns)
    return float(np.max(np.abs(U_q - U_e)))


# ---------------------------------------------------------------------------
# Plot (only used from __main__)
# ---------------------------------------------------------------------------

def plot_fidelity_sweep(
    results: list[GateFidelityResult], save_path: str | None = "fidelity_vs_noise.png",
    show: bool = True,
):
    import matplotlib.pyplot as plt

    x = np.array([r.params["noise_rms"] for r in results])
    y = np.array([r.infidelity for r in results])
    e = np.array([r.infidelity_sem for r in results])
    est = np.array([r.params["infidelity_static_estimate"] for r in results])
    u = results[0].params["noise_units"].replace("uV", "µV")
    tg = results[0].params["t_gate_ns"]

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.errorbar(x, y, yerr=e, fmt="o", color="tab:blue", capsize=3, label="Simulated (direct)")
    ax.plot(x, est, "--", color="tab:blue", lw=1.2, label=r"quasi-static estimate $t_g^2/(3T_{2,\mathrm{static}}^{*2})$")
    for thr, label, color in [(1e-2, "99 %", "tab:orange"), (1e-3, "99.9 %", "tab:green"),
                              (1e-4, "99.99 %", "tab:purple")]:
        ax.axhline(thr, color=color, ls=":", lw=1.3)
        ax.text(x.min(), thr * 1.15, label, color=color, fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Charge-noise RMS ({u})")
    ax.set_ylabel(r"Average gate infidelity $1-F$")
    ax.set_title(f"Single-gate infidelity vs charge noise ($t_{{\\rm gate}}$ = {tg:g} ns)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
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
    NOISE_UNITS = "uV"
    NOISE_VALUES = [0.5, 1, 2, 5, 10, 20, 30, 40, 50]      # noise RMS sweep
    T_GATE = T_GATE_NS

    print("Verification 1: noise-free fidelity")
    print(f"  F = {noise_free_fidelity(T_GATE):.10f}   (expected 1)\n")

    print("Verification 2: QuTiP propagator vs exact step product (one realization)")
    print(f"  max |U_qutip - U_exact| = {propagator_check(t_gate_ns=T_GATE):.2e}\n")

    print("Verification 3: single-point run at 10 µV")
    print(gate_fidelity_simulation(10.0, NOISE_UNITS, t_gate_ns=T_GATE))
    print()

    print("Sweep:")
    results = fidelity_sweep(NOISE_VALUES, noise_units=NOISE_UNITS, t_gate_ns=T_GATE)
    print(f"{'RMS (µV)':>9} {'1-F (sim)':>12} {'± SEM':>10} {'1-F (est.)':>12} {'sim/est':>8} {'F (%)':>9}")
    for r in results:
        p = r.params
        print(f"{p['noise_rms']:9.3g} {r.infidelity:12.3e} {r.infidelity_sem:10.1e} "
              f"{p['infidelity_static_estimate']:12.3e} {r.infidelity / p['infidelity_static_estimate']:8.2f} "
              f"{100 * r.F_avg:9.4f}")

    plot_fidelity_sweep(results)