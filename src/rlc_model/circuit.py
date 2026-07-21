"""Circuit model of one IRS element (Eq. 3 and Eq. 4 of the extension brief).

Every function is fully vectorized: the component arguments may be scalars or
numpy arrays of any (broadcastable) shape, e.g. (n_particles, N).
"""
import numpy as np

from rlc_model.config import OMEGA, Z0, BETA_MIN, K_PARAM, PHI_PARAM


def element_impedance(L1, L2, C, R, omega=OMEGA):
    """Equivalent impedance Z_n of the parallel resonant circuit (Eq. 3).

        Zn = jwL1 * (jwL2 + 1/(jwC) + R) / (jwL1 + jwL2 + 1/(jwC) + R)
    """
    Zb = 1j * omega * L2 + 1.0 / (1j * omega * C) + R   # series branch
    Za = 1j * omega * L1                                # parallel inductor
    return (Za * Zb) / (Za + Zb)


def reflection_coefficient(L1, L2, C, R, omega=OMEGA, z0=Z0):
    """Physical reflection coefficient v_n = (Zn - Z0) / (Zn + Z0) (Eq. 4).

    Returns a complex array with |v_n| <= 1 (passive circuit).
    """
    Zn = element_impedance(L1, L2, C, R, omega)
    return (Zn - z0) / (Zn + z0)


def eq5_amplitude(theta, beta_min=BETA_MIN, k=K_PARAM, phi=PHI_PARAM):
    """Equivalent-model amplitude beta(theta) of Eq. (5) (paper's fit).

    Used by the AO baseline and by exp1 to verify that the circuit model
    reproduces this curve.
    """
    return (1 - beta_min) * ((np.sin(theta - phi) + 1) / 2) ** k + beta_min


def eq5_reflection(theta):
    """Equivalent-model reflection coefficient v = beta(theta) * exp(j theta)."""
    return eq5_amplitude(theta) * np.exp(1j * theta)
