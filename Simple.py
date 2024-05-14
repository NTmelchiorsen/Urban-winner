import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.special import roots_hermite
from scipy.linalg import eig

def setup():
    par = {}
    # 1. demographics
    par['T'] = 40
    
    # 2. preferences
    par['rho'] = 0.75
    par['beta'] = 0.95
    par['alpha'] = 1.5
    par['lw'] = 1.8
    
    # 3. labour choices and human capital accumulation
    par['h'] = np.array([0, 0.5, 1])
    par['Nh'] = len(par['h'])
    par['delta'] = 0.1
    par['phi_1'] = 0.2
    par['phi_2'] = 0.6
    
    # 4. taste shocks
    par['sigma_eta'] = 0.25  # taste shocks
    
    # 5. income
    par['tax_rate'] = 0.0
    par['kappa'] = 1
    par['sigma_qå'] = 0.1  # capital uncertainty
    par['UB'] = 0.0
    
    # 6. saving
    par['R'] = 1.05
    
    # 7. grids and numerical integration
    par['m_max'] = 20.0
    par['m_phi'] = 1.1  # curvature parameters
    par['m_points_low'] = 50
    par['a_max'] = 20.0
    par['a_phi'] = 1.1  # curvature parameters
    par['k_max'] = 10.0
    par['k_phi'] = 1.0  # curvature parameters
    
    par['Nqå'] = 8
    par['Nm'] = 200
    par['Na'] = 200
    par['Nk'] = 150
    
    return par

def create_grids(par):
    # 1. Check parameters
    assert par['rho'] >= 0, 'rho must be greater than 0'
    assert par['Nm'] == par['Na'], 'Nm should equal Na'
    
    # Shocks
    par['qå'], par['qå_w'] = GaussHermite_lognorm(par['sigma_qå'], par['Nqå'])
    
    # End-of-period assets
    par['grid_a'] = [nonlinspace(1e-6, par['a_max'], par['Na'] * 8, par['a_phi']).reshape((par['Na'], 8)) for _ in range(par['T'])]
    
    # Cash-on-hand
    par['grid_m'] = nonlinspace(1e-4, par['m_max'], par['Nm'], par['m_phi'])
    
    # Human capital
    par['grid_k'] = nonlinspace(1e-4, par['k_max'], par['Nk'], par['k_phi'])
    
    return par

def GaussHermite_lognorm(sigma, n):
    h_nodes, h_weights = roots_hermite(n)
    x = np.exp(h_nodes * np.sqrt(2) * sigma - 0.5 * sigma**2)  # Adjust for lognormal distribution
    w = h_weights / np.sqrt(np.pi)
    assert np.abs(1 - np.sum(w * x)) < 1e-8  # Ensuring the mean is 1
    return x, w

def nonlinspace(lo, hi, n, phi):
    x = np.empty(n)
    x[0] = lo
    for i in range(1, n):
        x[i] = x[i - 1] + (hi - x[i - 1]) / ((n - i + 1) ** phi)
    return x


def solve(par):
    # 1. Allocate solution dictionary
    sol = {'c': {}, 'v': {}, 'v_plus': {}, 'm': {}}
    
    # 2. Last period (consume all)
    for i_nh in range(par['Nh']):
        sol['m'][(par['T'], i_nh)] = np.tile(par['grid_m'], (par['Nk'], 1)).T
        sol['c'][(par['T'], i_nh)] = np.tile(par['grid_m'], (par['Nk'], 1)).T
        sol['v'][(par['T'], i_nh)] = utility(sol['c'][(par['T'], i_nh)], i_nh, par)

    # 3. Before last period
    c_plus_interp = [None] * par['Nh']
    v_plus_interp = [None] * par['Nh']
    
    for t in range(par['T'] - 1, 0, -1):
        print(f'Solving period {t}')
        for i_nh in range(par['Nh']):
            sol['c'][(t, i_nh)] = np.nan * np.ones((par['Nm'], par['Nk']))
            sol['v'][(t, i_nh)] = np.nan * np.ones((par['Nm'], par['Nk']))
        
        # b. Create interpolants
        for i_nh in range(par['Nh']):
            c_plus_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['c'][(t + 1, i_nh)], method='linear')
            v_plus_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['v'][(t + 1, i_nh)], method='linear')
        
        # c. Solve by EGM
        for i_nh in range(par['Nh']):
            for i_k in range(par['Nk']):
                k = par['grid_k'][i_k]
                c_raw, m_raw, v_plus_raw = egm(t, i_nh, k, v_plus_interp, c_plus_interp, par)
                c, v = upper_envelope(c_raw, m_raw, v_plus_raw, i_nh, t, par)
                
                # Store solution
                sol['c'][(t, i_nh)][:, i_k] = c
                sol['v'][(t, i_nh)][:, i_k] = v
    return sol

def utility(c, h_choice, par):
    h_hour = par['h'][h_choice]
    return c ** (1 - par['rho']) / (1 - par['rho']) - par['lw'] * h_hour ** par['alpha'] / par['alpha']

def egm(t, h_choice, k, v_plus_interp, c_plus_interp, par):
    a = par['grid_a'][t]
    k = np.tile(k, (par['Na'], 1))  # Expand k across asset points
    w = np.tile(par['qå_w'], (par['Na'], 1))  # Ensure weights are tiled similarly
    qå = np.tile(par['qå'], (par['Na'], 1))  # Make sure qå is correctly tiled

    k_plus = k_trans(k, h_choice, qå, par)  # Ensure operations here don't alter shapes unexpectedly
    m_plus = m_trans(a, k, k_plus, h_choice, par)

    # Debug output shapes
    print("Shape of k_plus:", k_plus.shape)
    print("Shape of m_plus:", m_plus.shape)

    v_plus_vec_raw, prob = taste_exp(m_plus, k_plus, v_plus_interp, par)
    v_plus_raw = np.sum(w * v_plus_vec_raw, axis=1)  # Check this operation too

    avg_marg_u_plus = future_marg_u(c_plus_interp, m_plus, k_plus, prob, w, par)
    c_raw = inv_marg_utility_c(par['beta'] * par['R'] * avg_marg_u_plus, par)
    m_raw = a + c_raw

    return c_raw, m_raw, v_plus_raw


def k_trans(k, h_choice, qå, par):
    h_hour = par['h'][h_choice]
    return ((1 - par['delta']) * k + par['phi_1'] * h_hour ** par['phi_2']) * qå


def m_trans(a, k, k_plus, h_choice, par):
    h_hour = par['h'][h_choice]
    k = np.tile(k, (1, k_plus.shape[1]))
    return par['R'] * a + par['kappa'] * (1 - par['tax_rate']) * h_hour * k

def taste_exp(m_plus, k_plus, v_plus_interp, par):
    v_matrix = np.zeros((np.prod(m_plus.shape), par['Nh']))
    for i_nh in range(par['Nh']):  # Loop over labor choices
        indices = np.stack([m_plus.ravel(), k_plus.ravel()], axis=1)
        v_matrix[:, i_nh] = v_plus_interp[i_nh](indices).reshape(m_plus.shape)
    V_plus, prob = logsum_vec(v_matrix, par['sigma_eta'])
    return V_plus.reshape(m_plus.shape), prob