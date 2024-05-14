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
    par['sigma_xi'] = 0.1  # capital uncertainty
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
    
    par['Nxi'] = 8
    par['Nm'] = 200
    par['Na'] = 200
    par['Nk'] = 150
    
    return par

def create_grids(par):
    # 1. Check parameters
    assert par['rho'] >= 0, 'rho must be greater than 0'
    assert par['Nm'] == par['Na'], 'Nm should equal Na'
    
    # Shocks
    par['xi'], par['xi_w'] = GaussHermite_lognorm(par['sigma_xi'], par['Nxi'])
    
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

def printfig(figin):
    fig = plt.figure(figin)
    fig.set_size_inches(16/2.54, 12/2.54)  # Convert inches to centimeters
    filename = f"figs/{fig.get_label()}.pdf"
    fig.savefig(filename, format='pdf')

def logsum(v1, v2, sigma):
    V = np.stack([v1, v2], axis=1)
    if np.abs(sigma) > 1e-10:
        mxm = np.max(V, axis=1, keepdims=True)
        log_sum = mxm + sigma * np.log(np.sum(np.exp((V - mxm) / sigma), axis=1, keepdims=True))
        prob = np.exp((V - log_sum) / sigma)
    else:
        mxm = np.max(V, axis=1)
        prob = (V == mxm[:, None]).astype(float)
    return log_sum.flatten(), prob

def logsum_vec(V, sigma):
    if np.abs(sigma) > 1e-10:
        mxm = np.max(V, axis=1, keepdims=True)
        log_sum = mxm + sigma * np.log(np.sum(np.exp((V - mxm) / sigma), axis=1, keepdims=True))
        prob = np.exp((V - log_sum) / sigma)
    else:
        mxm = np.max(V, axis=1, keepdims=True)
        prob = (V == mxm).astype(float)
        log_sum = mxm
    return log_sum.flatten(), prob

def utility(c, h_choice, par):
    h_hour = par['h'][h_choice]
    return c ** (1 - par['rho']) / (1 - par['rho']) - par['lw'] * h_hour ** par['alpha'] / par['alpha']

def marg_utility_c(c, par):
    return c ** (-par['rho'])

def inv_marg_utility_c(u, par):
    return u ** (-1 / par['rho'])

def m_trans(a, k, k_plus, h_choice, par):
    h_hour = par['h'][h_choice]
    k = np.tile(k, (1, k_plus.shape[1]))
    return par['R'] * a + par['kappa'] * (1 - par['tax_rate']) * h_hour * k

def k_trans(k, h_choice, xi, par):
    h_hour = par['h'][h_choice]
    return ((1 - par['delta']) * k + par['phi_1'] * h_hour ** par['phi_2']) * xi

def taste_exp(m_plus, k_plus, v_plus_interp, par):
    v_matrix = np.zeros((np.prod(m_plus.shape), par['Nh']))
    for i_nh in range(par['Nh']):  # Loop over labor choices
        indices = np.stack([m_plus.ravel(), k_plus.ravel()], axis=1)
        v_matrix[:, i_nh] = v_plus_interp[i_nh](indices).reshape(m_plus.shape)
    V_plus, prob = logsum_vec(v_matrix, par['sigma_eta'])
    return V_plus.reshape(m_plus.shape), prob

def future_marg_u(c_plus_interp, m_plus, k_plus, prob, w, par):
    c_plus = []
    for i_nh in range(par['Nh']):
        indices = np.stack([m_plus.ravel(), k_plus.ravel()], axis=1)
        c_plus.append(c_plus_interp[i_nh](indices).reshape(m_plus.shape))
    marg_u_plus_matrix = np.array([marg_utility_c(c, par) for c in c_plus]).T
    marg_u_plus_taste = np.sum(prob * marg_u_plus_matrix, axis=1)
    return np.sum(w * marg_u_plus_taste.reshape(m_plus.shape), axis=1)

def value_of_choice_gridsearch(C, h, mt, kt, last, par):
    if last:  # Last period
        return utility(C, h, par)
    else:
        K_plus = ((1 - par['delta']) * kt + par['phi_1'] * par['h'][h] ** par['phi_2']) * par['xi']
        kt = np.tile(kt, (K_plus.shape[0], 1))
        M_plus = par['R'] * (mt - C) + par['kappa'] * par['h'][h] * kt + (par['h'][h] == 0) * par['UB']
        V1 = par['v_plus_interp'][0](np.stack([M_plus.ravel(), K_plus.ravel()], axis=1))
        V2 = par['v_plus_interp'][1](np.stack([M_plus.ravel(), K_plus.ravel()], axis=1))
        V3 = par['v_plus_interp'][2](np.stack([M_plus.ravel(), K_plus.ravel()], axis=1))
        return utility(C, h, par) + np.sum(par['xi_w'] * par['beta'] * logsum_2(V1, V2, V3, par['sigma_eta']))


def egm(t, h_choice, k, v_plus_interp, c_plus_interp, par):
    a = par['grid_a'][t]
    k = np.tile(k, (par['Na'], 1))  # Expand k across asset points
    w = np.tile(par['xi_w'], (par['Na'], 1))  # Ensure weights are tiled similarly
    xi = np.tile(par['xi'], (par['Na'], 1))  # Make sure xi is correctly tiled

    k_plus = k_trans(k, h_choice, xi, par)  # Ensure operations here don't alter shapes unexpectedly
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

    
def upper_envelope(c_raw, m_raw, v_plus_raw, h_choice, t, par):
    c_raw = np.insert(c_raw, 0, 1e-6)
    m_raw = np.insert(m_raw, 0, 1e-6)
    a_raw = np.insert(par['grid_a'][t], 0, 0)
    v_plus_raw = np.insert(v_plus_raw, 0, v_plus_raw[0])

    c = np.full(par['Nm'], np.nan)
    v = np.full(par['Nm'], -np.inf)

    for i in range(len(m_raw) - 1):
        m_low, m_high = m_raw[i], m_raw[i+1]
        c_slope = (c_raw[i+1] - c_raw[i]) / (m_high - m_low)

        a_low, a_high = a_raw[i], a_raw[i+1]
        v_plus_slope = (v_plus_raw[i+1] - v_plus_raw[i]) / (a_high - a_low)

        for j in range(par['Nm']):
            m_now = par['grid_m'][j]
            if m_low <= m_now <= m_high or (m_now > m_high and i == len(m_raw) - 2):
                c_guess = c_raw[i] + c_slope * (m_now - m_low)
                a_guess = m_now - c_guess
                v_plus = v_plus_raw[i] + v_plus_slope * (a_guess - a_low)
                v_guess = utility(c_guess, h_choice, par) + par['beta'] * v_plus
                if v_guess > v[j]:
                    v[j] = v_guess
                    c[j] = c_guess
    return c, v

def gridsearch(par, h, last):
    V = np.zeros((par['Nm'], par['Nk']))
    Cstar = np.full((par['Nm'], par['Nk']), np.nan)
    par['grid_C'] = np.linspace(0, 1, par['Nc'])

    for i_M in range(par['Nm']):
        mt = par['grid_m'][i_M]

        for i_K in range(par['Nk']):
            kt = par['grid_k'][i_K]

            for ic in range(par['Nc']):
                C = par['grid_C'][ic] * mt

                # a. find value of choice
                V_new = value_of_choice_gridsearch(C, h, mt, kt, last, par)

                # b. save if V_new > V
                if V_new > V[i_M, i_K]:
                    V[i_M, i_K] = V_new
                    Cstar[i_M, i_K] = C
    return V, Cstar

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
    
def sol_dif_pars(par, par_name, par_grid, N, m_ini, k_ini, seed):
    # 1. Initialize
    par['prefix'] = par_name
    
    # Simulation settings
    sim = {'N': N, 'm_ini': m_ini, 'k_ini': k_ini}
    
    # Store results
    store = {'par_grid': par_grid, 'par': [], 'sol': [], 'sim': []}
    
    # 2. Solve and simulate
    for i, val in enumerate(par_grid):
        par[par_name] = val  # Overwrite parameter
        store['par'].append(dict(par))  # Store current parameters
        
        # Solve model
        store['sol'].append(solve(par))
        
        # Simulate model
        store['sim'].append(simulate(sim, store['sol'][i], par, seed))
    
    return store

def sol_dif_pars_tax(par, par_name, par_grid, N, m_ini, k_ini, seed, time):
    # 1. Initialize
    par['prefix'] = par_name
    
    # Simulation settings
    sim = {'N': N, 'm_ini': m_ini, 'k_ini': k_ini}
    
    # Store results
    store = {'par_grid': par_grid, 'par': [], 'sol': [], 'sim': []}
    
    # 2. Solve and simulate
    for i, val in enumerate(par_grid):
        par[par_name] = val  # Overwrite parameter
        store['par'].append(dict(par))  # Store current parameters
        
        # Solve model
        store['sol'].append(solve(par))
    
    # Simulate models under different tax scenarios
    store['sim'].append(simulate(sim, store['sol'][0], par, seed))
    store['sim'].append(simulate_tax(sim, store['sol'][0], store['par'][0], store['sol'][1], store['par'][1], seed, time))
    
    return store

def solve_gridsearch(par):
    # 1. C-grid, M-grid and H-grid
    par['grid_C'] = np.linspace(0, 1, par['Nc'])
    
    # 2. Allocate memory
    sol = {'c': {}, 'v': {}}
    
    par['v_plus_interp'] = [None] * par['Nh']
    
    # 3. Last period
    for h in range(par['Nh']):
        sol['v'][(par['T'], h)], sol['c'][(par['T'], h)] = gridsearch(par, h, last=True)
    
    # 4. Backwards over time
    for t in range(par['T'] - 1, 0, -1):
        print(f'Solving period {t}')
        # a. Interpolant
        for h in range(par['Nh']):
            par['v_plus_interp'][h] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['v'][(t + 1, h)], method='linear')
        
        # b. Find V for all discrete choices and states
        for h in range(par['Nh']):
            sol['v'][(t, h)], sol['c'][(t, h)] = gridsearch(par, h, last=False)
    
    return sol, par

def euler_error(sim):
    abs_error = np.abs(sim['lhs_euler'] - sim['rhs_euler'])

    abs_error_0 = abs_error + (abs_error == 0)  # Set zeros equal to 1
    log10_abs_error_0 = np.log10(abs_error_0 / sim['c'][:, :-1])  # Include all zeros (log(1) = 0)

    abs_error_nan = np.where(abs_error == 0, np.nan, abs_error)  # Set zeros to NaN
    log10_abs_error_nan = np.log10(abs_error_nan / sim['c'][:, :-1])  # Exclude all zeros

    sim['euler_error'] = np.nanmean(abs_error_nan)
    sim['log10_euler_error'] = np.nanmean(log10_abs_error_0)
    sim['log10_euler_error_using_nan'] = np.nanmean(log10_abs_error_nan)

def simulate(sim, sol, par, seed):
    np.random.seed(seed)
    sim['m'] = np.full((sim['N'], par['T']), sim['m_ini'])
    sim['k'] = np.full((sim['N'], par['T']), sim['k_ini'])

    sim['c'] = np.full((sim['N'], par['T']), np.nan)
    sim['h_choice'] = np.full((sim['N'], par['T']), np.nan)
    sim['a'] = np.zeros((sim['N'], par['T']))
    sim['lhs_euler'] = np.full((sim['N'], par['T'] - 1), np.nan)
    sim['rhs_euler'] = np.full((sim['N'], par['T'] - 1), np.nan)

    unif = np.random.rand(sim['N'], par['T'])
    shock = np.exp(np.random.randn(sim['N'], par['T'] - 1) * par['sigma_xi'])

    for t in range(par['T']):
        v_matrix = np.full((sim['m'].shape[0], par['Nh']), np.nan)
        c_interp = [None] * par['Nh']
        c_plus_interp = [None] * par['Nh']

        for i_nh in range(par['Nh']):
            v_interp = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['v'][t, i_nh], method='linear')
            v_matrix[:, i_nh] = v_interp(np.column_stack((sim['m'][:, t], sim['k'][:, t])))
            c_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['c'][t, i_nh], method='linear')
            if t < par['T'] - 1:
                c_plus_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['c'][t + 1, i_nh], method='linear')

        _, prob = logsum_vec(v_matrix, par['sigma_eta'])

        prob_cum = np.cumsum(prob, axis=1)
        I = np.sum(unif[:, t, None] > prob_cum, axis=1) + 1
        sim['h_choice'][:, t] = I

        for i_nh in range(par['Nh']):
            ind = (I == (i_nh + 1))
            sim['c'][ind, t] = c_interp[i_nh](np.column_stack((sim['m'][ind, t], sim['k'][ind, t])))

        if t < par['T'] - 1:
            sim['a'][:, t] = sim['m'][:, t] - sim['c'][:, t]
            sim['k'][:, t + 1] = k_trans(sim['k'][:, t], I, shock[:, t], par)
            sim['m'][:, t + 1] = m_trans(sim['a'][:, t], sim['k'][:, t], sim['k'][:, t + 1], I, par)
            avg_marg_u_plus = future_marg_u(c_plus_interp, sim['m'][:, t + 1], sim['k'][:, t + 1], prob, 1, par)
            sim['lhs_euler'][:, t] = marg_utility_c(sim['c'][:, t], par)
            sim['rhs_euler'][:, t] = par['beta'] * par['R'] * avg_marg_u_plus

    # Calculate statistics after simulation
    sim['labor'] = np.array([np.mean(sim['h_choice'] == i + 1, axis=0) for i in range(par['Nh'])]).T
    sim['means'] = {
        'hours': np.sum(par['h'] * sim['labor'], axis=1),
        'cons': np.mean(sim['c'], axis=0),
        'assets': np.mean(sim['a'], axis=0),
        'cash': np.mean(sim['m'], axis=0),
        'capital': np.mean(sim['k'], axis=0),
        'wage': np.mean(par['kappa'] * sim['k'], axis=0)
    }
    sim = euler_error(sim)

    return sim
    
def simulate_tax(sim, sol_old, par_old, sol_tax, par_tax, seed, time):
    np.random.seed(seed)
    sim['m'] = np.full((sim['N'], par_old['T']), sim['m_ini'])
    sim['k'] = np.full((sim['N'], par_old['T']), sim['k_ini'])
    
    sim['c'] = np.nan * np.ones((sim['N'], par_old['T']))
    sim['h_choice'] = np.nan * np.ones((sim['N'], par_old['T']))
    sim['a'] = np.zeros((sim['N'], par_old['T']))
    sim['lhs_euler'] = np.nan * np.ones((sim['N'], par_old['T'] - 1))
    sim['rhs_euler'] = np.nan * np.ones((sim['N'], par_old['T'] - 1))
    
    unif = np.random.rand(sim['N'], par_old['T'])
    shock = np.exp(np.random.randn(sim['N'], par_old['T'] - 1) * par_old['sigma_xi'])
    
    # Simulate before tax change
    for t in range(time - 1):
        par = par_old
        sol = sol_old
        sim = simulate_step(sim, sol, par, t, unif, shock)
    
    # Simulate after tax change
    for t in range(time - 1, par_tax['T']):
        par = par_tax
        sol = sol_tax
        sim = simulate_step(sim, sol, par, t, unif, shock)
    
    # Calculate moments/statistics
    sim = calculate_statistics(sim, par)
    return sim

def simulate_step(sim, sol, par, t, unif, shock):
    v_matrix = np.full((sim['m'].shape[0], par['Nh']), np.nan)
    c_interp = [None] * par['Nh']
    c_plus_interp = [None] * par['Nh']
    
    for i_nh in range(par['Nh']):
        v_interp = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['v'][t, i_nh], method='linear')
        v_matrix[:, i_nh] = v_interp(np.column_stack((sim['m'][:, t], sim['k'][:, t])))
        c_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['c'][t, i_nh], method='linear')
        if t < par['T'] - 1:
            c_plus_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['c'][t + 1, i_nh], method='linear')
    
    _, prob = logsum_vec(v_matrix, par['sigma_eta'])
    prob_cum = np.cumsum(prob, axis=1)
    I = np.sum(unif[:, t, None] > prob_cum, axis=1) + 1
    sim['h_choice'][:, t] = I

    for i_nh in range(par['Nh']):
        ind = (I == (i_nh + 1))
        sim['c'][ind, t] = c_interp[i_nh](np.column_stack((sim['m'][ind, t], sim['k'][ind, t])))

    # Update states
    if t < par['T'] - 1:
        sim['a'][:, t] = sim['m'][:, t] - sim['c'][:, t]
        sim['k'][:, t + 1] = k_trans(sim['k'][:, t], I, shock[:, t], par)
        sim['m'][:, t + 1] = m_trans(sim['a'][:, t], sim['k'][:, t], sim['k'][:, t + 1], I, par)
        avg_marg_u_plus = future_marg_u(c_plus_interp, sim['m'][:, t + 1], sim['k'][:, t + 1], prob, 1, par)
        sim['lhs_euler'][:, t] = marg_utility_c(sim['c'][:, t], par)
        sim['rhs_euler'][:, t] = par['beta'] * par['R'] * avg_marg_u_plus
    
    return sim

def calculate_statistics(sim, par):
    # Labor supply statistics
    sim['labor'] = np.nan * np.ones((par['Nh'], par['T']))
    for i in range(par['Nh']):
        sim['labor'][i, :] = np.mean(sim['h_choice'] == i + 1, axis=0)
    
    # Average hours, consumption, assets, cash on hand, human capital, wages
    sim['means'] = {
        'hours': np.sum(par['h'] * sim['labor'], axis=0),
        'cons': np.mean(sim['c'], axis=0),
        'assets': np.mean(sim['a'], axis=0),
        'cash': np.mean(sim['m'], axis=0),
        'capital': np.mean(sim['k'], axis=0),
        'wage': np.mean(par['kappa'] * sim['k'], axis=0)
    }
    
    # Euler errors
    sim = euler_error(sim)
    return sim

def reaction(tax):
    # Calculate percentage change in after-tax rate
    after_tax = ((1 - tax['post_tax']) / (1 - tax['pre_tax']) - 1) * 100
    # Average labor supply before and after tax change
    pre_supply = np.mean(tax['pre_supply'])
    post_supply = np.mean(tax['post_supply'])
    # Calculate elasticity
    elasticity = (post_supply / pre_supply - 1) * 100 / after_tax
    return elasticity

def perm_tax_time(pre_tax, post_tax, par, seed, time):
    # Setup tax struct
    tax = {}
    tax['pre_tax'] = pre_tax
    tax['post_tax'] = post_tax
    
    # Setup for solution and simulation
    par_name = 'tax_rate'
    par_grid = [pre_tax, post_tax]
    N = 10**5
    m_ini = 1.5
    k_ini = 1
    
    # Store solutions and simulations
    store = sol_dif_pars_tax(par, par_name, par_grid, N, m_ini, k_ini, seed, time)
    # Extract labor supply before and after tax change
    tax['pre_supply'] = par['h'][store['sim'][1]['h_choice'][:, :par['T'] - 1]]
    tax['post_supply'] = par['h'][store['sim'][2]['h_choice'][:, :par['T'] - 1]]
    
    return tax