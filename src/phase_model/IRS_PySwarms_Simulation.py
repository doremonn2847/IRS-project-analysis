import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

M = 2             # Number of AP antennas
N = 40            # Number of IRS reflecting elements
Pt_dBm = 36       # Total transmit power at AP (dBm)
sigma2_dBm = -94  # Noise power (dBm)

# Convert dBm to linear scale (Watts)
Pt = 10**((Pt_dBm - 30) / 10)
sigma2 = 10**((sigma2_dBm - 30) / 10)

# Practical phase shift model parameters
beta_min = 0.2
k = 1.6
phi = 0.43 * np.pi

# Path loss parameters
PL_ref = 10**(-40 / 10) # 40 dB attenuation at 1m reference distance
alpha_AI = 2.2  # AP-IRS
alpha_IU = 2.8  # IRS-User
alpha_AU = 3.8  # AP-User

def generateRayleighChannels(num_elements, num_antennas, dist_d):
    dist_AI = 500.0
    dist_IU = np.sqrt((500.0 - dist_d) ** 2 + 4.0)
    dist_AU = np.sqrt(dist_d ** 2 + 4.0)
    pl_ai = PL_ref * (500.0 ** (-2.2))
    pl_au = PL_ref * (dist_AU ** (-3.8))
    pl_iu = PL_ref * (dist_IU ** (-2.8))
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

def runSimulation():
    dist_d = 498.0
    print(f"Generating channels for distance: {dist_d}m")
    phi_mat, channel_hd = generateRayleighChannels(N, M, dist_d)
    
    # Define bounds for theta [-pi, pi]
    max_bound = np.ones(N) * np.pi
    min_bound = -max_bound
    bounds = (min_bound, max_bound)
    
    # Set up pyswarms options
    # c1: cognitive parameter, c2: social parameter, w: inertia weight
    options = {'c1': 2.05, 'c2': 2.05, 'w': 0.729}
    
    # Create the optimizer
    optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=N, options=options, bounds=bounds)
    
    # Create a wrapper for the objective function since PySwarms passes a 2D array of particles
    def cost_wrapper(x):
        n_particles = x.shape[0]
        costs = np.zeros(n_particles)
        for i in range(n_particles):
            costs[i] = objectiveFunctionPractical(x[i], phi_mat, channel_hd, Pt, sigma2, beta_min, k, phi)
        return costs

    # Perform optimization
    print("Running PySwarms optimization...")
    cost, pos = optimizer.optimize(cost_wrapper, iters=150)
    
    best_rate = -cost
    print(f"Optimization complete!")
    print(f"Best Achievable Rate: {best_rate:.4f} bps/Hz")
    
    return best_rate, pos

runSimulation()