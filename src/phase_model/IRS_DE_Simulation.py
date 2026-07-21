import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# IEEE publication-quality styling (auto-applied on import)
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))  # so 'ieee_plot_style' (in src/) is importable
import ieee_plot_style
from ieee_plot_style import IEEE_COLORS

def generateRayleighChannels(num_elements, num_antennas, dist_d):
    pl_zero = 10.0 ** (-4.0)
    dist_au = np.sqrt(dist_d ** 2 + 4.0)
    dist_iu = np.sqrt((500.0 - dist_d) ** 2 + 4.0)
    pl_ai = pl_zero * (500.0 ** (-2.2))
    pl_au = pl_zero * (dist_au ** (-3.8))
    pl_iu = pl_zero * (dist_iu ** (-2.8))
    channel_g = np.sqrt(pl_ai / 2.0) * (np.random.randn(num_elements, num_antennas) + 1j * np.random.randn(num_elements, num_antennas))
    channel_hr = np.sqrt(pl_iu / 2.0) * (np.random.randn(num_elements, 1) + 1j * np.random.randn(num_elements, 1))
    channel_hd = np.sqrt(pl_au / 2.0) * (np.random.randn(num_antennas, 1) + 1j * np.random.randn(num_antennas, 1))
    matrix_phi = np.dot(np.diag(channel_hr.conj().flatten()), channel_g)
    return matrix_phi, channel_hd

def computeReflectionVector(theta_vec, beta_min, k_param, phi_param):
    amplitude = (1.0 - beta_min) * ((np.sin(theta_vec - phi_param) + 1.0) / 2.0) ** k_param + beta_min
    return amplitude * np.exp(1j * theta_vec)

def objectiveFunctionPractical(theta_vec, phi_mat, hd_vec, p_t, sigma_sq, beta_min, k_param, phi_param):
    v_vec = computeReflectionVector(theta_vec, beta_min, k_param, phi_param)
    combined_channel = np.dot(v_vec.conj().T, phi_mat) + hd_vec.conj().T
    channel_gain = np.linalg.norm(combined_channel) ** 2
    achievable_rate = np.log2(1.0 + (p_t * channel_gain) / sigma_sq)
    return -achievable_rate

def objectiveFunctionIdeal(theta_vec, phi_mat, hd_vec, p_t, sigma_sq):
    v_vec = np.exp(1j * theta_vec)
    combined_channel = np.dot(v_vec.conj().T, phi_mat) + hd_vec.conj().T
    channel_gain = np.linalg.norm(combined_channel) ** 2
    achievable_rate = np.log2(1.0 + (p_t * channel_gain) / sigma_sq)
    return -achievable_rate

def solveAoProposition1(phi_mat, hd_vec, p_t, sigma_sq, beta_min, k_param, phi_param):
    num_elements = 40
    theta_vec = np.random.choice([np.pi, -np.pi], size=num_elements)
    v_vec = computeReflectionVector(theta_vec, beta_min, k_param, phi_param)
    for iteration in range(10):
        for n in range(num_elements):
            combined_channel = np.dot(v_vec.conj().T, phi_mat) + hd_vec.conj().T
            row_n = phi_mat[n, :]
            vector_a = combined_channel - v_vec[n].conj() * row_n
            phi_n_val = np.sum(row_n * vector_a.conj())
            arg_phi_n = np.angle(phi_n_val)
            lambda_val = 0 if arg_phi_n >= 0 else 1
            theta_a = arg_phi_n
            theta_c = ((-1.0) ** lambda_val) * np.pi
            theta_b = (theta_a + theta_c) / 2.0
            v_n_a = ((1.0 - beta_min) * ((np.sin(theta_a - phi_param) + 1.0) / 2.0) ** k_param + beta_min) * np.exp(1j * theta_a)
            f_1 = np.linalg.norm(vector_a + v_n_a.conj() * row_n) ** 2
            v_n_b = ((1.0 - beta_min) * ((np.sin(theta_b - phi_param) + 1.0) / 2.0) ** k_param + beta_min) * np.exp(1j * theta_b)
            f_2 = np.linalg.norm(vector_a + v_n_b.conj() * row_n) ** 2
            v_n_c = ((1.0 - beta_min) * ((np.sin(theta_c - phi_param) + 1.0) / 2.0) ** k_param + beta_min) * np.exp(1j * theta_c)
            f_3 = np.linalg.norm(vector_a + v_n_c.conj() * row_n) ** 2
            denom = 4.0 * (f_1 - 2.0 * f_2 + f_3)
            if np.abs(denom) > 1e-6:
                theta_opt = (theta_c * (3.0 * f_1 - 4.0 * f_2 + f_3) + theta_a * (f_1 - 4.0 * f_2 + 3.0 * f_3)) / denom
            else:
                theta_opt = theta_a
            theta_opt = np.clip(theta_opt, -np.pi, np.pi)
            theta_vec[n] = theta_opt
            v_vec[n] = ((1.0 - beta_min) * ((np.sin(theta_opt - phi_param) + 1.0) / 2.0) ** k_param + beta_min) * np.exp(1j * theta_opt)
    combined_channel = np.dot(v_vec.conj().T, phi_mat) + hd_vec.conj().T
    channel_gain = np.linalg.norm(combined_channel) ** 2
    return np.log2(1.0 + (p_t * channel_gain) / sigma_sq)

def runSimulation():
    num_elements = 40
    num_antennas = 2
    p_t = 10.0 ** ((36.0 - 30.0) / 10.0)
    sigma_sq = 10.0 ** ((-94.0 - 30.0) / 10.0)
    beta_min = 0.2
    k_param = 1.6
    phi_param = 0.43 * np.pi
    d_steps = np.array([480, 485, 490, 495, 498, 500])
    num_monte_carlo = 5
    rates_practical = np.zeros(len(d_steps))
    rates_ideal_assump = np.zeros(len(d_steps))
    rates_no_irs = np.zeros(len(d_steps))
    rates_ao = np.zeros(len(d_steps))
    bounds_list = [(-np.pi, np.pi)] * num_elements
    for mc in range(num_monte_carlo):
        for idx, dist_d in enumerate(d_steps):
            phi_mat, hd_vec = generateRayleighChannels(num_elements, num_antennas, dist_d)
            res_practical = differential_evolution(
                objectiveFunctionPractical, bounds_list,
                args=(phi_mat, hd_vec, p_t, sigma_sq, beta_min, k_param, phi_param),
                strategy='best1bin', maxiter=100, popsize=10, tol=0.05
            )
            rates_practical[idx] += -res_practical.fun
            res_ideal = differential_evolution(
                objectiveFunctionIdeal, bounds_list,
                args=(phi_mat, hd_vec, p_t, sigma_sq),
                strategy='best1bin', maxiter=100, popsize=10, tol=0.05
            )
            actual_rate_ideal = -objectiveFunctionPractical(res_ideal.x, phi_mat, hd_vec, p_t, sigma_sq, beta_min, k_param, phi_param)
            rates_ideal_assump[idx] += actual_rate_ideal
            gain_no_irs = np.linalg.norm(hd_vec) ** 2
            rates_no_irs[idx] += np.log2(1.0 + (p_t * gain_no_irs) / sigma_sq)
            rates_ao[idx] += solveAoProposition1(phi_mat, hd_vec, p_t, sigma_sq, beta_min, k_param, phi_param)
    rates_practical /= num_monte_carlo
    rates_ideal_assump /= num_monte_carlo
    rates_no_irs /= num_monte_carlo
    rates_ao /= num_monte_carlo
    fig, ax = ieee_plot_style.new_figure('single_tall')
    ax.plot(d_steps, rates_practical,    color=IEEE_COLORS['green'],
            linestyle='-', marker='o', markersize=4, label='Practical IRS (DE)')
    ax.plot(d_steps, rates_ideal_assump, color=IEEE_COLORS['red'],
            linestyle='--', marker='x', markersize=4, label='Practical IRS with Ideal Assumption')
    ax.plot(d_steps, rates_ao,           color=IEEE_COLORS['blue'],
            linestyle='-.', marker='D', markersize=4, label='Practical IRS (AO — Paper)')
    ax.plot(d_steps, rates_no_irs,       color=IEEE_COLORS['gray'],
            linestyle=':', marker='s', markersize=4, label='No IRS')
    ax.set_xlabel('AP-user horizontal distance $d$ (m)')
    ax.set_ylabel('Achievable rate (bits/s/Hz)')
    ieee_plot_style.format_legend(ax, loc='best')
    ieee_plot_style.save_figure('irs_simulation_result.png')
    plt.show()

# if __name__ == '_main_':
runSimulation()