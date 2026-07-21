"""Wireless system model: channels and achievable rate (Eq. 6).

Channel generation is identical to the weeks 1-4 scripts (Rayleigh fading +
distance-dependent path loss, Section V of the reference paper) so that the
component-level results are directly comparable with the phase-level results.
"""
import numpy as np

from rlc_model.config import (M, D_AP_IRS, D_PERP, C0, ALPHA_AI, ALPHA_IU,
                              ALPHA_AU, PT, SIGMA2)


def path_loss(distance, alpha):
    """Amplitude-domain path loss sqrt(C0 * d^-alpha)."""
    return np.sqrt(C0 * distance ** (-alpha))


def generate_channels(d_user, n_elements, rng):
    """One random realization of the three channels.

    Returns:
        hd:  (M,)   direct AP -> user channel
        hr:  (N,)   IRS -> user channel
        G:   (N, M) AP -> IRS channel
        Phi: (N, M) cascaded channel diag(hr^H) @ G  (precomputed)
    """
    d_irs_user = np.sqrt((d_user - D_AP_IRS) ** 2 + D_PERP ** 2)

    def cn(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)

    hd = path_loss(d_user, ALPHA_AU) * cn(M)
    hr = path_loss(d_irs_user, ALPHA_IU) * cn(n_elements)
    G = path_loss(D_AP_IRS, ALPHA_AI) * cn(n_elements, M)
    Phi = np.diag(hr.conj()) @ G
    return hd, hr, G, Phi


def rate_from_reflection(v, hd, Phi):
    """Achievable rate R_SE (Eq. 6) with optimal MRT beamforming.

    Vectorized over leading dimensions: v may be (N,) or (P, N); the return
    value is a scalar or a (P,) array respectively.
    """
    combined = v.conj() @ Phi + hd.conj()           # (..., M)
    signal_power = PT * np.sum(np.abs(combined) ** 2, axis=-1)
    return np.log2(1 + signal_power / SIGMA2)


def rate_no_irs(hd):
    """Lower bound: direct link only, no IRS."""
    return np.log2(1 + PT * np.linalg.norm(hd) ** 2 / SIGMA2)


def ideal_upper_bound(hd, Phi, n_sweeps=30):
    """Upper bound: ideal IRS with |v_n| = 1 and free phase (coordinate ascent).

    Returns (v_ideal, rate).
    """
    n_elements = Phi.shape[0]
    v = np.ones(n_elements, dtype=complex)
    for _ in range(n_sweeps):
        for n in range(n_elements):
            combined = v.conj() @ Phi + hd.conj()
            contrib = Phi[n, :]
            without_n = combined - v[n].conj() * contrib
            v[n] = np.exp(1j * np.angle(contrib @ without_n.conj()))
    return v, rate_from_reflection(v, hd, Phi)
