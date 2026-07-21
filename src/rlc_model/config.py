"""Central configuration: system parameters, circuit constants, component bounds.

System-level parameters are identical to the weeks 1-4 simulations
(Section V of the reference paper) so results remain directly comparable.
"""
import pathlib

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SRC_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
FIGURES_DIR = REPO_ROOT / "results" / "figures"
DATA_DIR = REPO_ROOT / "results" / "data"

# ── System model (same as weeks 1-4) ────────────────────────────────────────
M = 2                      # number of AP antennas
N = 40                     # number of IRS reflecting elements (default)
D_AP_IRS = 500             # AP-IRS horizontal distance (m)
D_PERP = 2                 # perpendicular offset of the user line (m)

C0 = 10 ** (-40 / 10)      # reference path loss at 1 m (linear)
ALPHA_AI = 2.2             # AP-IRS path loss exponent
ALPHA_IU = 2.8             # IRS-user path loss exponent
ALPHA_AU = 3.8             # AP-user (direct) path loss exponent

PT_DBM = 36
NOISE_DBM = -94
PT = 10 ** ((PT_DBM - 30) / 10)          # transmit power (W)
SIGMA2 = 10 ** ((NOISE_DBM - 30) / 10)   # noise power (W)

# ── Circuit constants (Eq. 3-4 of the extension brief) ──────────────────────
FREQ = 2.4e9               # operating frequency (Hz)
OMEGA = 2 * np.pi * FREQ   # angular frequency (rad/s)
Z0 = 377.0                 # free-space impedance (Ohm)

# Fixed component values used in the reference paper (Abeywickrama et al.):
# bottom-layer inductance L1 = 2.5 nH, top-layer inductance L2 = 0.7 nH,
# effective loss resistance R = 2.5 Ohm. With these values, sweeping the
# varactor capacitance C reproduces the equivalent model of Eq. (5) with
# beta_min = 0.2, k = 1.6, phi = 0.43*pi (verified in exp1).
L1_PAPER = 2.5e-9
L2_PAPER = 0.7e-9
R_PAPER = 2.5

# Equivalent phase-shift model constants (Eq. 5, used by the AO baseline)
BETA_MIN = 0.2
K_PARAM = 1.6
PHI_PARAM = 0.43 * np.pi

# ── Physical component bounds (Eq. 6-9 of the extension brief) ──────────────
# Justification (stated in the report as required):
#  * C:  0.47-2.35 pF is the tuning range of the varactor diode used in the
#        reference paper (SMV1231-style device, reverse bias 0-15 V).
#  * L1: paper value 2.5 nH; range covers typical chip/patch inductances
#        realizable at 2.4 GHz (0.5-5 nH).
#  * L2: paper value 0.7 nH; range covers the parasitic/top-layer
#        inductance of practical element layouts (0.1-2 nH).
#  * R:  paper studies R in {1, 2.5} Ohm as the effective series loss of the
#        varactor + metal; 0.5 Ohm is a realistic lower loss floor, 5 Ohm a
#        pessimistic upper bound.
COMPONENT_NAMES = ("L1", "L2", "C", "R")
BOUNDS = {
    "L1": (0.5e-9, 5.0e-9),
    "L2": (0.1e-9, 2.0e-9),
    "C": (0.47e-12, 2.35e-12),
    "R": (0.5, 5.0),
}
PAPER_VALUES = {"L1": L1_PAPER, "L2": L2_PAPER, "C": None, "R": R_PAPER}

# Display scales for figures/tables (value / scale -> label)
COMPONENT_SCALES = {"L1": (1e-9, "nH"), "L2": (1e-9, "nH"),
                    "C": (1e-12, "pF"), "R": (1.0, "Ohm")}


def ensure_output_dirs():
    """Create results/figures and results/data if missing."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
