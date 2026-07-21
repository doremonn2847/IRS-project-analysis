"""Week-5 extension: component-level (RLC) optimization of IRS elements.

Instead of optimizing the abstract phase shift theta_n (equivalent model
v_n = beta(theta_n) * exp(j*theta_n), Eq. (5) of the reference paper), the
optimizer works directly on the physical circuit parameters of every IRS
element: L1_n, L2_n, C_n, R_n.

Pipeline per candidate solution x = [L1_1, L2_1, C_1, R_1, ..., R_N]:
    components -> Z_n (Eq. 3) -> v_n = (Z_n - Z0)/(Z_n + Z0) (Eq. 4)
               -> R_SE = log2(1 + |(v^H Phi + hd^H) w|^2 / sigma^2) (Eq. 6)
"""
