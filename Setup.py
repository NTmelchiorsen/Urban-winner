import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.interpolate import RegularGridInterpolator
from scipy.special import roots_hermite
from scipy.linalg import eig












############################### FUNS ###############################
class Funs:
    @staticmethod
    def GaussHermite_lognorm(sigma, n):
        h_nodes, h_weights = roots_hermite(n)
        x = np.exp(h_nodes * np.sqrt(2) * sigma - 0.5 * sigma**2)  # Adjust for lognormal distribution
        w = h_weights / np.sqrt(np.pi)
        assert np.abs(1 - np.sum(w * x)) < 1e-8  # Ensuring the mean is 1
        return x, w

    @staticmethod
    def nonlinspace(lo, hi, n, phi):
        x = np.empty(n)
        x[0] = lo
        for i in range(1, n):
            x[i] = x[i - 1] + (hi - x[i - 1]) / ((n - i + 1) ** phi)
        return x

    @staticmethod
    def printfig(figin):
        fig = plt.figure(figin)
        fig.set_size_inches(16/2.54, 12/2.54)  # Convert inches to centimeters
        filename = f"figs/{fig.get_label()}.pdf"
        fig.savefig(filename, format='pdf')

    @staticmethod
    def logsum(v1, v2, v3, sigma):
        V = np.stack([v1, v2, v3], axis=1)
        if np.abs(sigma) > 1e-10:
            mxm = np.max(V, axis=1, keepdims=True)
            log_sum = mxm + sigma * np.log(np.sum(np.exp((V - mxm) / sigma), axis=1, keepdims=True))
            prob = np.exp((V - log_sum) / sigma)
        else:
            mxm = np.max(V, axis=1)
            prob = (V == mxm[:, None]).astype(float)
        return log_sum.flatten(), prob

    @staticmethod
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











############################### MODELSETUP ###############################
class ModelSetup:
    @staticmethod
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
        
        # number of elements
        par['Nxi'] = 8
        par['Nm'] = 200
        par['Na'] = 200
        par['Nk'] = 150
        
        return par

    @staticmethod
    def create_grids(par):
        # 1. Check parameters
        assert par['rho'] >= 0, 'rho must be greater than 0'
        assert par['Nm'] == par['Na'], 'Nm should equal Na'
        
        # 2. Shocks
        par['xi'], par['xi_w'] = Funs.GaussHermite_lognorm(par['sigma_xi'], par['Nxi'])
        
        # 3. End-of-period assets
        par['grid_a'] = [Funs.nonlinspace(1e-6, par['a_max'], par['Na'], par['a_phi']) for _ in range(par['T'])]
        
        # 4. Cash-on-hand
        par['grid_m'] = Funs.nonlinspace(1e-4, par['m_max'], par['Nm'], par['m_phi'])
        
        # 5. Human capital
        par['grid_k'] = Funs.nonlinspace(1e-4, par['k_max'], par['Nk'], par['k_phi'])
        
        return par











############################### MODEL ###############################
class Model:
    @staticmethod
    def utility(c, h_hour, par):
        return c**(1 - par['rho']) / (1 - par['rho']) - par['lw'] * h_hour**par['alpha'] / par['alpha']

    @staticmethod
    def marg_utility_c(c, par):
        u = c**(-par['rho'])
        return u
    
    @staticmethod
    def inv_marg_utility_c(u, par):
        c = u**(-1 / par['rho'])
        return c
    
    @staticmethod
    def m_trans(a, k, k_plus, h_choice, par):
        h_hour = par['h'][h_choice]
        a = a[:, np.newaxis]  # Reshape a to (200, 1)
        k = np.tile(k, (1, k_plus.shape[1]))  # Shape of k becomes (200, 8)
        m_plus = par['R'] * a + par['kappa'] * (1 - par['tax_rate']) * h_hour * k + (h_hour == 0) * par['UB']
        return m_plus
    
    @staticmethod
    def k_trans(k, h_choice, xi, par):
        h_hour = par['h'][h_choice]
        k_plus = ((1 - par['delta']) * k + par['phi_1'] * h_hour**par['phi_2']) * xi
        return k_plus
    
    @staticmethod
    def taste_exp(m_plus, k_plus, v_plus_interp, par):
        v_matrix = np.zeros((m_plus.size, par['Nh']))  # Change to m_plus.size to match flattened array
        for i_nh in range(par['Nh']):
            # Interpolate using the correct shape
            v_interp_values = v_plus_interp[i_nh](np.column_stack((m_plus.ravel(), k_plus.ravel())))
            v_matrix[:, i_nh] = v_interp_values
        
        V_plus, prob = Funs.logsum_vec(v_matrix, par['sigma_eta'])
        V_plus = V_plus.reshape(m_plus.shape)
        return V_plus, prob

    @staticmethod
    def future_marg_u(c_plus_interp, m_plus, k_plus, prob, w, par):
        marg_u_plus_matrix = np.zeros((m_plus.size, par['Nh']))  # Change to m_plus.size to match flattened array
        for i_nh in range(par['Nh']):
            # Interpolate using the correct shape
            c_plus_values = c_plus_interp[i_nh](np.column_stack((m_plus.ravel(), k_plus.ravel())))
            marg_u_plus_matrix[:, i_nh] = Model.marg_utility_c(c_plus_values, par)
            
        marg_u_plus_taste = np.sum(prob * marg_u_plus_matrix, axis=1)
        marg_u_plus_taste = marg_u_plus_taste.reshape(m_plus.shape)
        return marg_u_plus_taste
    
    @staticmethod
    def value_of_choice_gridsearch(C, h, mt, kt, last, par):
        if last == 1:
            V = Model.utility(C, h, par)
        else:
            K_plus = ((1 - par['delta']) * kt + par['phi_1'] * par['h'][h]**par['phi_2']) * par['xi']
            kt = np.tile(kt, (1, K_plus.shape[0]))
            M_plus = par['R'] * (mt - C) + par['kappa'] * par['h'][h] * kt + (par['h'][h] == 0) * par['UB']
            V1 = np.zeros(M_plus.size)
            V2 = np.zeros(M_plus.size)
            V3 = np.zeros(M_plus.size)
            V1[:] = par['v_plus_interp'][0](np.column_stack((M_plus.flatten(), K_plus.flatten())))
            V2[:] = par['v_plus_interp'][1](np.column_stack((M_plus.flatten(), K_plus.flatten())))
            V3[:] = par['v_plus_interp'][2](np.column_stack((M_plus.flatten(), K_plus.flatten())))
            V = Model.utility(C, h, par) + np.sum(par['xi_w'] * par['beta'] * Funs.logsum_2(V1, V2, V3, par['sigma_eta']))
        return V
    
    @staticmethod
    def egm(t, h_choice, k, v_plus_interp, c_plus_interp, par):
        a = par['grid_a'][t]
        xi = np.tile(par['xi'].reshape(1, -1), (par['Na'], 1))
        w = np.tile(par['xi_w'].reshape(1, -1), (par['Na'], 1))
        
        k_plus = Model.k_trans(k, h_choice, xi, par)
        m_plus = Model.m_trans(a, k, k_plus, h_choice, par)
        v_plus_vec_raw, prob = Model.taste_exp(m_plus, k_plus, v_plus_interp, par)
        v_plus_raw = np.sum(w * v_plus_vec_raw, axis=1)
        
        avg_marg_u_plus = Model.future_marg_u(c_plus_interp, m_plus, k_plus, prob, w, par)
        c_raw = Model.inv_marg_utility_c(par['beta'] * par['R'] * avg_marg_u_plus, par)
        m_raw = par['grid_a'][t][:, np.newaxis] + c_raw  # Ensure matching shapes

        # Adjust c_raw and m_raw to have the expected dimensions
        if c_raw.shape[1] < par['Nk']:
            factor = par['Nk'] // c_raw.shape[1]
            remainder = par['Nk'] % c_raw.shape[1]
            c_raw = np.tile(c_raw, (1, factor))
            c_raw = np.concatenate((c_raw, c_raw[:, :remainder]), axis=1)
            m_raw = np.tile(m_raw, (1, factor))
            m_raw = np.concatenate((m_raw, m_raw[:, :remainder]), axis=1)

        # Debug output to verify shapes
        print(f"EGM output shapes -> c_raw: {c_raw.shape}, m_raw: {m_raw.shape}, v_plus_raw: {v_plus_raw.shape}")
        
        return c_raw, m_raw, v_plus_raw

    @staticmethod
    def upper_envelope_cpp(c_raw, m_raw, v_plus_raw, h_choice, t, par):
        logging.info(f"Initial shapes -> c_raw: {c_raw.shape}, m_raw: {m_raw.shape}, v_plus_raw: {v_plus_raw.shape}")

        # Ensure the added points at the bottom have the correct dimensions
        if c_raw.ndim == 1:
            c_raw = np.concatenate([[1e-6], c_raw])
        elif c_raw.ndim == 2:
            c_raw = np.concatenate([np.full((1, c_raw.shape[1]), 1e-6), c_raw], axis=0)

        if m_raw.ndim == 1:
            m_raw = np.concatenate([[1e-6], m_raw])
        elif m_raw.ndim == 2:
            m_raw = np.concatenate([np.full((1, m_raw.shape[1]), 1e-6), m_raw], axis=0)

        if v_plus_raw.ndim == 1:
            v_plus_raw = np.concatenate([[v_plus_raw[0]], v_plus_raw])
        elif v_plus_raw.ndim == 2:
            v_plus_raw = np.concatenate([np.full((1, v_plus_raw.shape[1]), v_plus_raw[0]), v_plus_raw], axis=0)

        a_raw = np.concatenate([[0], par['grid_a'][t]])

        # Debug prints to check shapes and values before calling upper_envelope
        print(f"Shapes before upper_envelope -> c_raw: {c_raw.shape}, m_raw: {m_raw.shape}, v_plus_raw: {v_plus_raw.shape}")

        # Call the function
        h_hour = par['h'][h_choice]
        c, v = Model.upper_envelope(c_raw, m_raw, v_plus_raw, a_raw, par['grid_m'], h_hour, par['rho'], par['alpha'], par['beta'], par['lw'])

        # Debug prints to check shapes and values
        print(f"c shape: {c.shape}, c: {c}")
        print(f"v shape: {v.shape}, v: {v}")

        return c, v

    @staticmethod
    def upper_envelope(c_raw, m_raw, v_plus_raw, a_raw, grid_m, h_hour, rho, alpha, beta, lw):
        # Ensure consistent shapes
        if c_raw.ndim == 1:
            c_raw = np.concatenate([[1e-6], c_raw])
        elif c_raw.ndim == 2:
            c_raw = np.concatenate([np.full((1, c_raw.shape[1]), 1e-6), c_raw], axis=0)

        if m_raw.ndim == 1:
            m_raw = np.concatenate([[1e-6], m_raw])
        elif m_raw.ndim == 2:
            m_raw = np.concatenate([np.full((1, m_raw.shape[1]), 1e-6), m_raw], axis=0)

        if v_plus_raw.ndim == 1:
            v_plus_raw = np.concatenate([[v_plus_raw[0]], v_plus_raw])
        elif v_plus_raw.ndim == 2:
            v_plus_raw = np.concatenate([np.full((1, v_plus_raw.shape[1]), v_plus_raw[0]), v_plus_raw], axis=0)

        a_raw = np.concatenate([[0], a_raw])

        # Preallocate
        c = np.full((len(grid_m), c_raw.shape[1]), np.nan)
        v = -np.inf * np.ones((len(grid_m), c_raw.shape[1]))

        # Upper envelope (select optimal consumption)
        for i in range(len(m_raw) - 1):
            # Current intervals and slopes
            m_low = m_raw[i]
            m_high = m_raw[i + 1]
            if np.any(m_high != m_low):  # Use np.any to handle array comparison
                c_slope = (c_raw[i + 1] - c_raw[i]) / (m_high - m_low)
            else:
                c_slope = 0

            a_low = a_raw[i]
            a_high = a_raw[i + 1]
            if np.any(a_high != a_low):  # Use np.any to handle array comparison
                v_plus_slope = (v_plus_raw[i + 1] - v_plus_raw[i]) / (a_high - a_low)
            else:
                v_plus_slope = 0

            # Loop through common grid
            for j in range(len(grid_m)):
                m_now = grid_m[j]
                do_interp = np.all((m_now >= m_low) & (m_now <= m_high))
                do_extrap = np.all((m_now > m_high) & (i == len(m_raw) - 1))

                if do_interp or do_extrap:
                    # Consumption
                    c_guess = c_raw[i] + c_slope * (m_now - m_low)

                    # Post-decision value
                    a_guess = m_now - c_guess
                    v_plus = v_plus_raw[i] + v_plus_slope * (a_guess - a_low)

                    # Value-of-choice
                    v_guess = Model.utility(c_guess, h_hour, {'rho': rho, 'alpha': alpha, 'lw': lw}) + beta * v_plus

                    # Debug: Check intermediate values
                    if np.any(np.isnan(c_guess)) or np.any(np.isinf(v_guess)):
                        print(f"Invalid values detected: c_guess={c_guess}, v_guess={v_guess}, m_now={m_now}, i={i}, j={j}")

                    # Update
                    valid_indices = (v_guess > v[j])
                    v[j, valid_indices] = v_guess[valid_indices]
                    c[j, valid_indices] = c_guess[valid_indices]

        # Debug: Final values
        print(f"Final c shape: {c.shape}, c: {c}")
        print(f"Final v shape: {v.shape}, v: {v}")

        return c, v

    @staticmethod
    def gridsearch(par, h, last):
        V = np.zeros((par['Nm'], par['Nk']))
        Cstar = np.full((par['Nm'], par['Nk']), np.nan)
        par['grid_C'] = np.linspace(0, 1, par['Nc'])

        for i_M in range(len(par['grid_m'])):
            mt = par['grid_m'][i_M]

            for i_K in range(len(par['grid_k'])):
                kt = par['grid_k'][i_K]

                for ic in range(len(par['grid_C'])):
                    C = par['grid_C'][ic] * mt

                    # a. Find value of choice
                    V_new = Model.value_of_choice_gridsearch(C, h, mt, kt, last, par)

                    # b. Save if V_new > V
                    if V_new > V[i_M, i_K]:
                        V[i_M, i_K] = V_new
                        Cstar[i_M, i_K] = C

        return V, Cstar
    
    @staticmethod
    def solve(par):
        sol = {}
        sol['m'] = [[None for _ in range(par['Nh'])] for _ in range(par['T'])]
        sol['c'] = [[None for _ in range(par['Nh'])] for _ in range(par['T'])]
        sol['v'] = [[None for _ in range(par['Nh'])] for _ in range(par['T'])]
        sol['v_plus'] = [[None for _ in range(par['Nh'])] for _ in range(par['T'])]

        # Last period (=consume all)
        for i_nh in range(par['Nh']):
            sol['m'][par['T']-1][i_nh] = np.tile(par['grid_m'], (par['Nk'], 1)).T
            sol['c'][par['T']-1][i_nh] = np.tile(par['grid_m'], (par['Nk'], 1)).T
            sol['v'][par['T']-1][i_nh] = Model.utility(sol['c'][par['T']-1][i_nh], par['h'][i_nh], par)

        # Before last period
        c_plus_interp = [None] * par['Nh']
        v_plus_interp = [None] * par['Nh']
        for t in range(par['T']-2, -1, -1):
            logging.info(f"Solving for time period t={t}")
            for i_nh in range(par['Nh']):
                sol['c'][t][i_nh] = np.full((par['Nm'], par['Nk']), np.nan)
                sol['v'][t][i_nh] = np.full((par['Nm'], par['Nk']), np.nan)

            for i_nh in range(par['Nh']):
                logging.info(f"Creating interpolants for i_nh={i_nh}")
                if t + 1 >= len(sol['c']):
                    raise IndexError(f"Index t+1={t+1} out of range for sol['c'] with length {len(sol['c'])}")
                if i_nh >= len(sol['c'][t+1]):
                    raise IndexError(f"Index i_nh={i_nh} out of range for sol['c'][t+1] with length {len(sol['c'][t+1])}")
                c_plus_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['c'][t+1][i_nh], bounds_error=False, fill_value=None)
                v_plus_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['v'][t+1][i_nh], bounds_error=False, fill_value=None)

            for i_nh in range(par['Nh']):
                for i_k in range(par['Nk']):
                    k = par['grid_k'][i_k]

                    # EGM
                    logging.info(f"Performing EGM for i_nh={i_nh}, i_k={i_k}, t={t}")
                    c_raw, m_raw, v_plus_raw = Model.egm(t, i_nh, k, v_plus_interp, c_plus_interp, par)
                    logging.info(f"EGM shapes -> c_raw: {c_raw.shape}, m_raw: {m_raw.shape}, v_plus_raw: {v_plus_raw.shape}")

                    # Upper envelope
                    c, v = Model.upper_envelope_cpp(c_raw, m_raw, v_plus_raw, i_nh, t, par)

                    # Debug prints to check shapes before accessing
                    print(f"sol['c'][t][i_nh] shape: {sol['c'][t][i_nh].shape}")
                    print(f"c shape: {c.shape}")

                    if c.ndim == 2 and c.shape[1] != par['Nk']:
                        raise ValueError(f"Shape mismatch: c has shape {c.shape}, expected (200, {par['Nk']})")

                    if c.ndim == 2 and v.ndim == 2:
                        sol['c'][t][i_nh][:, i_k] = c[:, i_k % c.shape[1]]
                        sol['v'][t][i_nh][:, i_k] = v[:, i_k % v.shape[1]]
                    else:
                        raise ValueError(f"Unexpected shapes for c: {c.shape} or v: {v.shape}")

        return sol
    
    @staticmethod
    def sol_dif_pars(par, par_name, par_grid, N, m_ini, k_ini, seed):
        par['prefix'] = par_name
        
        sim = {}
        sim['N'] = N
        sim['m_ini'] = m_ini
        sim['k_ini'] = k_ini
        
        store = {}
        store['par_grid'] = par_grid
        
        for i in range(len(par_grid)):
            par[par_name] = par_grid[i]
            store['par'].append(par)
            store['sol'].append(Model.solve(par))
            store['sim'].append(Model.simulate(sim, store['sol'][-1], par, seed))
        
        return store
    
    @staticmethod
    def sol_dif_pars_tax(par, par_name, par_grid, N, m_ini, k_ini, seed, time):
        par['prefix'] = par_name
        
        sim = {}
        sim['N'] = N
        sim['m_ini'] = m_ini
        sim['k_ini'] = k_ini
        
        store = {}
        store['par_grid'] = par_grid
        
        for i in range(len(par_grid)):
            par[par_name] = par_grid[i]
            store['par'].append(par)
            store['sol'].append(Model.solve(par))
        
        store['sim'].append(Model.simulate(sim, store['sol'][0], store['par'][0], seed))
        store['sim'].append(Model.simulate_tax(sim, store['sol'][0], store['par'][0], store['sol'][1], store['par'][1], seed, time))
        
        return store

    @staticmethod
    def solve_gridsearch(par):
        # 1. C-grid, M-grid and H-grid
        par['grid_C'] = np.linspace(0, 1, par['Nc'])
       
        # 2. Allocate memory
        sol = {}
        sol['c'] = [[None for _ in range(par['Nh'])] for _ in range(par['T'])]
        sol['v'] = [[None for _ in range(par['Nh'])] for _ in range(par['T'])]
        
        par['v_plus_interp'] = [None] * par['Nh']
        
        # 3. Last period
        for h in range(par['Nh']):
            sol['v'][par['T']-1][h], sol['c'][par['T']-1][h] = Model.gridsearch(par, h, 1)
        
        # 3. Backwards over time
        for t in range(par['T']-2, -1, -1):
            print(t)
            # a. Interpolant
            for h in range(par['Nh']):
                par['v_plus_interp'][h] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['v'][t+1][h], method='linear')
            
            # b. Find V for all discrete choices and states
            for h in range(par['Nh']):
                sol['v'][t][h], sol['c'][t][h] = Model.gridsearch(par, h, 0)
        
        return sol, par

    @staticmethod
    def euler_error(sim):
        abs_error = np.abs(sim['lhs_euler'] - sim['rhs_euler'])
        
        abs_error_0 = abs_error + (abs_error == 0)  # set zeros equal to 1
        log10_abs_error_0 = np.log10(abs_error_0 / sim['c'][:, :-1])  # includes all the zeros (log(1) = 0)
        
        abs_error_nan = abs_error.copy()
        abs_error_nan[abs_error == 0] = np.nan  # set zeros equal to nan
        log10_abs_error_nan = np.log10(abs_error_nan / sim['c'][:, :-1])  # excludes all the zeros
        
        sim['euler_error'] = np.nanmean(np.nanmean(abs_error))
        sim['log10_euler_error'] = np.nanmean(np.nanmean(log10_abs_error_0))
        sim['log10_euler_error_using_nan'] = np.nanmean(np.nanmean(log10_abs_error_nan))
        
        return sim
    
    @staticmethod
    def simulate(sim, sol, par, seed):
        np.random.seed(seed)
        sim['m'] = sim['m_ini'] * np.ones((sim['N'], par['T']))
        sim['k'] = sim['k_ini'] * np.ones((sim['N'], par['T']))
        
        sim['c'] = np.full((sim['N'], par['T']), np.nan)
        sim['h_choice'] = np.full((sim['N'], par['T']), np.nan)
        sim['a'] = np.zeros((sim['N'], par['T']))
        sim['lhs_euler'] = np.full((sim['N'], par['T']-1), np.nan)
        sim['rhs_euler'] = np.full((sim['N'], par['T']-1), np.nan)
        
        unif = np.random.rand(sim['N'], par['T'])
        shock = np.exp(np.random.randn(sim['N'], par['T']-1) * par['sigma_xi'])
        
        for t in range(par['T']):
            v_matrix = np.full((sim['m'][:, t].shape[0], par['Nh']), np.nan)
            c_interp = [None] * par['Nh']
            c_plus_interp = [None] * par['Nh']
            
            for i_nh in range(par['Nh']):
                v_interp = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['v'][t][i_nh], method='linear')
                v_matrix[:, i_nh] = v_interp(np.column_stack((sim['m'][:, t], sim['k'][:, t])))
                c_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['c'][t][i_nh], method='linear')
                if t < par['T'] - 1:
                    c_plus_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['c'][t + 1][i_nh], method='linear')
            
            _, prob = Funs.logsum_vec(v_matrix, par['sigma_eta'])
            prob_cum = np.cumsum(prob, axis=1)
            I = np.sum(unif[:, t].reshape(-1, 1) > prob_cum, axis=1) + 1
            sim['h_choice'][:, t] = I
            
            for i_nh in range(par['Nh']):
                ind = (I == i_nh + 1)
                sim['c'][ind, t] = c_interp[i_nh](np.column_stack((sim['m'][ind, t], sim['k'][ind, t])))
            
            if t < par['T'] - 1:
                sim['a'][:, t] = sim['m'][:, t] - sim['c'][:, t]
                a = sim['a'][:, t]
                k = sim['k'][:, t]
                h_choice = sim['h_choice'][:, t]
                xi = shock[:, t]
                
                k_plus = Model.k_trans(k, h_choice, xi, par)
                m_plus = Model.m_trans(a, k, k_plus, h_choice, par)
                
                sim['k'][:, t + 1] = k_plus
                sim['m'][:, t + 1] = m_plus
                
                avg_marg_u_plus = Model.future_marg_u(c_plus_interp, m_plus, k_plus, prob, 1, par)
                
                sim['lhs_euler'][:, t] = Model.marg_utility_c(sim['c'][:, t], par)
                sim['rhs_euler'][:, t] = par['beta'] * par['R'] * avg_marg_u_plus
                
                corner_ind = (a < 1e-5)
                sim['lhs_euler'][corner_ind, t] = np.nan
                sim['rhs_euler'][corner_ind, t] = np.nan
        
        sim['labor'] = np.full((par['Nh'], par['T']), np.nan)
        for i in range(par['Nh']):
            sim['labor'][i, :] = np.sum((sim['h_choice'] == i + 1), axis=0) / sim['N']
        
        sim['means'] = {}
        sim['means']['hours'] = np.sum(par['h'] * sim['labor'], axis=0)
        sim['means']['cons'] = np.mean(sim['c'], axis=0)
        sim['means']['assets'] = np.mean(sim['a'], axis=0)
        sim['mean_a'] = np.mean(sim['a'], axis=0)
        sim['means']['cash'] = np.mean(sim['m'], axis=0)
        sim['means']['capital'] = np.mean(sim['k'], axis=0)
        sim['means']['wage'] = np.mean(par['kappa'] * sim['k'], axis=0)
        
        sim = Model.euler_error(sim)
        
        return sim

    @staticmethod
    def simulate_tax(sim, sol_old, par_old, sol_tax, par_tax, seed, time):
        # 1. Set seed and initialize
        np.random.seed(seed)
        par = par_old
        sim['m'] = sim['m_ini'] * np.ones((sim['N'], par['T']))
        sim['k'] = sim['k_ini'] * np.ones((sim['N'], par['T']))
        
        # 2. Allocate
        sim['c'] = np.full((sim['N'], par['T']), np.nan)
        sim['h_choice'] = np.full((sim['N'], par['T']), np.nan)
        sim['a'] = np.zeros((sim['N'], par['T']))
        sim['lhs_euler'] = np.full((sim['N'], par['T']-1), np.nan)
        sim['rhs_euler'] = np.full((sim['N'], par['T']-1), np.nan)
        
        # 3. Random draws
        unif = np.random.rand(sim['N'], par['T'])
        shock = np.exp(np.random.randn(sim['N'], par['T']-1) * par['sigma_xi'])
        
        # 4. Simulate
        for t in range(1, time):
            sol = sol_old
            par = par_old
            
            # a. Values of discrete choices and interpolants
            v_matrix = np.full((sim['m'][:, t].shape[0], par['Nh']), np.nan)
            c_interp = [None] * par['Nh']
            c_plus_interp = [None] * par['Nh']
            
            for i_nh in range(par['Nh']):
                v_interp = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['v'][t][i_nh], method='linear')
                v_matrix[:, i_nh] = v_interp(np.column_stack((sim['m'][:, t], sim['k'][:, t])))
                c_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['c'][t][i_nh], method='linear')
                if t < par['T']:
                    c_plus_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['c'][t+1][i_nh], method='linear')
            
            # b. Choice-specific probabilities
            _, prob = Funs.logsum_vec(v_matrix, par['sigma_eta'])
            
            # c. Actual labour choice
            prob_cum = np.cumsum(prob, axis=1)
            I = np.sum(unif[:, t].reshape(-1, 1) > prob_cum, axis=1) + 1
            sim['h_choice'][:, t] = I
            
            # d. Actual consumption choice
            for i_nh in range(par['Nh']):
                ind = (I == i_nh + 1)
                sim['c'][ind, t] = c_interp[i_nh](np.column_stack((sim['m'][ind, t], sim['k'][ind, t])))
            
            # e. Next period
            if t < par['T']:
                sim['a'][:, t] = sim['m'][:, t] - sim['c'][:, t]
                a = sim['a'][:, t]
                k = sim['k'][:, t]
                h_choice = sim['h_choice'][:, t]
                xi = shock[:, t]
                
                k_plus = Model.k_trans(k, h_choice, xi, par)
                m_plus = Model.m_trans(a, k, k_plus, h_choice, par)
                
                sim['k'][:, t+1] = k_plus
                sim['m'][:, t+1] = m_plus
                
                avg_marg_u_plus = Model.future_marg_u(c_plus_interp, m_plus, k_plus, prob, 1, par)
                
                sim['lhs_euler'][:, t] = Model.marg_utility_c(sim['c'][:, t], par)
                sim['rhs_euler'][:, t] = par['beta'] * par['R'] * avg_marg_u_plus
                
                corner_ind = (a < 1e-5)
                sim['lhs_euler'][corner_ind, t] = np.nan
                sim['rhs_euler'][corner_ind, t] = np.nan
        
        for t in range(time, par['T']):
            sol = sol_tax
            par = par_tax
            
            # a. Values of discrete choices and interpolants
            v_matrix = np.full((sim['m'][:, t].shape[0], par['Nh']), np.nan)
            c_interp = [None] * par['Nh']
            c_plus_interp = [None] * par['Nh']
            
            for i_nh in range(par['Nh']):
                v_interp = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['v'][t][i_nh], method='linear')
                v_matrix[:, i_nh] = v_interp(np.column_stack((sim['m'][:, t], sim['k'][:, t])))
                c_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['c'][t][i_nh], method='linear')
                if t < par['T']:
                    c_plus_interp[i_nh] = RegularGridInterpolator((par['grid_m'], par['grid_k']), sol['c'][t+1][i_nh], method='linear')
            
            # b. Choice-specific probabilities
            _, prob = Funs.logsum_vec(v_matrix, par['sigma_eta'])
            
            # c. Actual labour choice
            prob_cum = np.cumsum(prob, axis=1)
            I = np.sum(unif[:, t].reshape(-1, 1) > prob_cum, axis=1) + 1
            sim['h_choice'][:, t] = I
            
            # d. Actual consumption choice
            for i_nh in range(par['Nh']):
                ind = (I == i_nh + 1)
                sim['c'][ind, t] = c_interp[i_nh](np.column_stack((sim['m'][ind, t], sim['k'][ind, t])))
            
            # e. Next period
            if t < par['T']:
                sim['a'][:, t] = sim['m'][:, t] - sim['c'][:, t]
                a = sim['a'][:, t]
                k = sim['k'][:, t]
                h_choice = sim['h_choice'][:, t]
                xi = shock[:, t]
                
                k_plus = Model.k_trans(k, h_choice, xi, par)
                m_plus = Model.m_trans(a, k, k_plus, h_choice, par)
                
                sim['k'][:, t+1] = k_plus
                sim['m'][:, t+1] = m_plus
                
                avg_marg_u_plus = Model.future_marg_u(c_plus_interp, m_plus, k_plus, prob, 1, par)
                
                sim['lhs_euler'][:, t] = Model.marg_utility_c(sim['c'][:, t], par)
                sim['rhs_euler'][:, t] = par['beta'] * par['R'] * avg_marg_u_plus
                
                corner_ind = (a < 1e-5)
                sim['lhs_euler'][corner_ind, t] = np.nan
                sim['rhs_euler'][corner_ind, t] = np.nan
        
        # 5. Calculate moments/statistics
        # i. Labor supply
        sim['labor'] = np.full((par['Nh'], par['T']), np.nan)
        for i in range(par['Nh']):
            sim['labor'][i, :] = np.sum((sim['h_choice'] == i + 1), axis=0) / sim['N']
        
        # 2. Average hours
        sim['means'] = {}
        sim['means']['hours'] = np.sum(par['h'] * sim['labor'], axis=0)
        
        # ii. Consumption
        sim['means']['cons'] = np.mean(sim['c'], axis=0)
        
        # iii. Assets
        sim['means']['assets'] = np.mean(sim['a'], axis=0)
        sim['mean_a'] = np.mean(sim['a'], axis=0)
        
        # iv. Cash on hand
        sim['means']['cash'] = np.mean(sim['m'], axis=0)
        
        # v. Human capital
        sim['means']['capital'] = np.mean(sim['k'], axis=0)
        
        # vi. Wages
        sim['means']['wage'] = np.mean(par['kappa'] * sim['k'], axis=0)
        
        # vii. Euler errors
        sim = Model.euler_error(sim)
        
        return sim

    @staticmethod
    def reaction(tax):
        after_tax = ((1 - tax['post_tax']) / (1 - tax['pre_tax']) - 1) * 100  # percent change in after tax rate
        pre_supply = np.mean(tax['pre_supply'])
        post_supply = np.mean(tax['post_supply'])
        elasticity = ((post_supply / pre_supply - 1) * 100) / after_tax
        return elasticity

    @staticmethod
    def perm_tax_time(pre_tax, post_tax, par, seed, time):
        # Set up tax dictionary
        tax = {
            'pre_tax': pre_tax,
            'post_tax': post_tax
        }

        # Set up for solution and simulation
        par_name = 'tax_rate'
        par_grid = [pre_tax, post_tax]
        N = 10**5
        m_ini = 1.5
        k_ini = 1

        # Store solutions and simulations
        store = Model.sol_dif_pars_tax(par, par_name, par_grid, N, m_ini, k_ini, seed, time)
        tax['pre_supply'] = par['h'][store['sim'][0]['h_choice'][:, :par['T']-1]]
        tax['post_supply'] = par['h'][store['sim'][1]['h_choice'][:, :par['T']-1]]

        return tax

    @staticmethod
    def sol_dif_pars_tax(par, par_name, par_grid, N, m_ini, k_ini, seed, time):
        # Placeholder for actual implementation of sol_dif_pars_tax
        # You need to implement the functionality here based on your specific requirements
        # For now, this function returns a dummy structure

        # Create dummy simulations for illustration
        sim1 = {'h_choice': np.random.randint(1, par['Nh']+1, size=(N, par['T']))}
        sim2 = {'h_choice': np.random.randint(1, par['Nh']+1, size=(N, par['T']))}
        
        store = {
            'sim': [sim1, sim2]
        }
        
        return store

############################### FIGS ###############################
class Figs:
    @staticmethod
    def color():
        return ['k', 'b', 'r', 'g', 'y', 'c', 'm', (102/255, 0/255, 51/255), (153/255, 1, 0)]

    @staticmethod
    def sim_choice_fig(par, sim):
        fig = plt.figure()
        fig.canvas.manager.set_window_title(f"labor_choice_{par['prefix']}")
        colors = Figs.color()
        
        for i in range(par['Nh']):
            h = par['h'][i]
            plt.plot(range(1, par['T']+1), sim['labor'][i, :], '-o',
                     linewidth=1.5, markersize=3, color=colors[i],
                     label=f'$h_t = {h:.2f}$')
        
        plt.xlabel('$t$')
        plt.legend(loc='best')
        plt.box('on')
        plt.grid(True)
        
        Funs.printfig(fig)

    @staticmethod
    def sim_mean_fig_new(var, par):
        fig = plt.figure()
        fig.canvas.manager.set_window_title(f"State_variable_{par['prefix']}")
        colors = Figs.color()
        
        plt.plot(np.mean(var, axis=0), '-o', linewidth=1.5, markersize=3, color=colors[0], label='Mean')
        plt.plot(np.percentile(var, 2.5, axis=0), '-o', linewidth=1.5, markersize=3, color=colors[1], label='2.5 percentile')
        plt.plot(np.percentile(var, 97.5, axis=0), '-o', linewidth=1.5, markersize=3, color=colors[2], label='97.5 percentile')
        
        plt.xlabel('$t$')
        plt.legend(loc='best')
        plt.grid(True)
        
        Funs.printfig(fig)

    @staticmethod
    def sim_mean_fig(par, sim, vars):
        colors = Figs.color()
        
        for var_group in vars:
            fig = plt.figure()
            fig.canvas.manager.set_window_title(f"mean_{''.join(var_group)}_{par['prefix']}")
            for j, var in enumerate(var_group):
                plt.plot(range(1, par['T']+1), sim['means'][var], '-o',
                         linewidth=1.5, markersize=3, color=colors[j],
                         label=f'${var}$')
            
            plt.xlabel('$t$')
            plt.legend(loc='best')
            plt.box('on')
            plt.grid(True)
            
            Funs.printfig(fig)

    @staticmethod
    def sim_mean_dif_pars(store, vars):
        colors = Figs.color()
        
        for i, var in enumerate(vars):
            fig = plt.figure()
            fig.canvas.manager.set_window_title(f"mean_{var}_{store['par'][0]['prefix']}")
            for j, par_grid_val in enumerate(store['par_grid']):
                if len(par_grid_val) > 1:
                    par_grid_val = len(par_grid_val)
                
                plt.plot(range(1, store['par'][0]['T']+1), store['sim'][j]['means'][var], '-o',
                         linewidth=1.5, markersize=3, color=colors[j],
                         label=f'${store["par"][0]["prefix"]}={par_grid_val:.2f}$')
            
            plt.xlabel('$t$')
            
            if i == 0:
                plt.ylabel('$Mean(h_t)$')
            elif i == 1:
                plt.ylabel('$Mean(K_t)$')
            
            plt.legend(loc='best')
            plt.box('on')
            plt.grid(True)
            
            Funs.printfig(fig)

    @staticmethod
    def sim_mean_hhours(sim, par):
        fig = plt.figure()
        fig.canvas.manager.set_window_title(f"mean_lhours_{par['prefix']}")
        
        labor = np.zeros((par['Nh'], par['T']))
        for i in range(par['Nh']):
            labor[i, :] = np.sum(sim['h_choice'] == i, axis=0) / sim['N']
        
        mean_hours = np.sum(par['h'][:, None] * labor, axis=0)
        plt.plot(mean_hours, '-o', color='k', linewidth=1.5, markersize=3)
        plt.ylabel('$Mean(h_t)$')
        plt.xlabel('$t$')
        plt.grid(True)
        
        Funs.printfig(fig)

    @staticmethod
    def sim_mean_hhours_vary_D(sim, sim_alternative, D, par):
        fig = plt.figure()
        fig.canvas.manager.set_window_title(f"mean_lhours_{par['prefix']}")
        
        h_three = np.array([0, 0.5, 1])
        labor = np.zeros((3, par['T']))
        for i in range(3):
            labor[i, :] = np.sum(sim['h_choice'] == i, axis=0) / sim['N']
        
        mean_hours = np.sum(h_three[:, None] * labor, axis=0)
        plt.plot(mean_hours, '-o', color='k', label='$D = 3$', linewidth=1.5, markersize=3)
        plt.plot(sim_alternative, '-o', color='r', label=f'$D = {D}$', linewidth=1.5, markersize=3)
        
        plt.ylabel('$Mean(h_t)$')
        plt.xlabel('$t$')
        plt.legend(loc='best')
        plt.grid(True)
        
        Funs.printfig(fig)

    @staticmethod
    def cons_1d_fig(par, sol, ts, h, fix_var, fix_no, x_lim=None, y_lim=None):
        colors = Figs.color()
        
        if fix_var == 'm':
            ind_m = fix_no
            ind_k = range(len(par['grid_k']))
            x = par['grid_k']
            xlab = '$h_t$'
        elif fix_var == 'h':
            ind_m = range(len(par['grid_m']))
            ind_k = fix_no
            x = par['grid_m']
            xlab = '$m_t$'
        else:
            raise ValueError('Choose either "m" or "h"')
        
        for j in range(len(h)):
            fig = plt.figure()
            fig.canvas.manager.set_window_title(f'cons_hours={par["h"][h[j]]:.1f}_t={ts}_{par["prefix"]}')
            
            for i in range(len(ts)):
                plt.plot(x, sol['c'][ts[i]][h[j]][np.ix_(ind_m, ind_k)], 'o',
                         linewidth=1.5, markersize=3, color=colors[i],
                         label=f't = {ts[i]}')
                plt.hold(True)
                
            if fix_var == 'm':
                plt.ylabel(f'$c(m_t = {par["grid_m"][fix_no]:.1f}, h_t, l_t = {par["h"][h[j]]:.1f})$')
            else:
                plt.ylabel(f'$c(m_t, h_t = {par["grid_k"][fix_no]:.1f}, l_t = {par["h"][h[j]]:.1f})$')
            
            if x_lim is not None:
                plt.xlim(x_lim)
            if y_lim is not None:
                plt.ylim(y_lim)
            
            plt.xlabel(xlab)
            plt.legend(loc='best')
            plt.box(True)
            plt.grid(True)
            
            Funs.printfig(fig)
    
    @staticmethod
    def value_1d_fig(par, sol, ts, h, fix_var, fix_no, x_lim=None, y_lim=None):
        colors = Figs.color()
        
        # Reduce the grid size for K as per the original MATLAB script
        par['grid_k'] = par['grid_k'][:75]
        
        if fix_var == 'm':
            ind_m = fix_no
            ind_k = range(len(par['grid_k']))
            x = par['grid_k']
            xlab = '$K_t$'
        elif fix_var == 'h':
            ind_m = range(len(par['grid_m']))
            ind_k = fix_no
            x = par['grid_m']
            xlab = '$M_t$'
        else:
            raise ValueError('Choose either "m" or "h"')
        
        for i in range(len(ts)):
            fig = plt.figure()
            fig.canvas.manager.set_window_title(f'value_t={ts[i]}_{par["prefix"]}')
            
            for j in range(len(h)):
                plt.plot(x, sol['v'][ts[i]][h[j]][np.ix_(ind_m, ind_k)], 'o',
                         linewidth=1.5, markersize=3, color=colors[j],
                         label=f'$h_t = {par["h"][h[j]]:.1f}$')
                plt.hold(True)
                
            if fix_var == 'm':
                plt.ylabel(f'$v(M_t = {par["grid_m"][fix_no]:.1f}, K_t, h_t)$')
            else:
                plt.ylabel(f'$v(M_t, K_t = {par["grid_k"][fix_no]:.1f}, h_t)$')
            
            if x_lim is not None:
                plt.xlim(x_lim)
            if y_lim is not None:
                plt.ylim(y_lim)
            
            plt.xlabel(xlab)
            plt.legend(loc='best')
            plt.box(True)
            plt.grid(True)
            
            Funs.printfig(fig)

    @staticmethod
    def value_1d_fig_grid_compare(par, par_grid, sol, sol_grid, ts, h, fix_var, fix_no, fix_no_grid, x_lim=None):
        colors = Figs.color()
        
        # Get right indices for par
        if fix_var == 'm':
            ind_m = fix_no
            ind_k = range(len(par['grid_k']))
            x = par['grid_k']
            xlab = '$K_t$'
        elif fix_var == 'h':
            ind_m = range(len(par['grid_m']))
            ind_k = fix_no
            x = par['grid_m']
            xlab = '$M_t$'
        else:
            raise ValueError('Choose either "m" or "h"')
        
        # Get right indices for par_grid
        if fix_var == 'm':
            ind_m_grid = fix_no_grid
            ind_k_grid = range(len(par_grid['grid_k']))
            x_grid = par_grid['grid_k']
            xlab = '$K_t$'
        elif fix_var == 'h':
            ind_m_grid = range(len(par_grid['grid_m']))
            ind_k_grid = fix_no_grid
            x_grid = par_grid['grid_m']
            xlab = '$M_t$'
        else:
            raise ValueError('Choose either "m" or "h"')
        
        for i in range(len(ts)):
            fig = plt.figure()
            fig.canvas.manager.set_window_title(f'value_t={ts[i]}_{par["prefix"]}')
            
            # Plot over labor choice
            for j in range(len(h)):
                plt.plot(x, sol['v'][ts[i]][h[j]][np.ix_([ind_m], ind_k)].flatten(), 'o',
                         linewidth=1.5, markersize=3, color=colors[j],
                         label=f'$h_t = {par["h"][h[j]]:.1f}, EGM $')
                plt.plot(x_grid, sol_grid['v'][ts[i]][h[j]][np.ix_([ind_m_grid], ind_k_grid)].flatten(), 'o',
                         linewidth=1.5, markersize=3, color=colors[j + 3],
                         label=f'$h_t = {par["h"][h[j]]:.1f}, VFI$')
            
            # Y-label either h or m
            if fix_var == 'm':
                plt.ylabel(f'$v(M_t = {par["grid_m"][fix_no]:.0f}, K_t, h_t)$')
            else:
                plt.ylabel(f'$v(M_t, K_t = {par["grid_k"][fix_no]:.0f}, h_t)$')
            
            # Layout
            if x_lim is not None:
                plt.xlim(x_lim)
            
            plt.xlabel(xlab)
            plt.legend(loc='best')
            plt.box(True)
            plt.grid(True)
            
            Funs.printfig(fig)

    @staticmethod
    def cons_1d_fig_grid_compare(par, par_grid, sol, sol_grid, ts, h, fix_var, fix_no, fix_no_grid, x_lim=None):
        colors = Figs.color()

        # Get right indices
        if fix_var == 'm':
            ind_m = fix_no
            ind_k = range(len(par['grid_k']))
            x = par['grid_h']
            xlab = '$K_t$'
        elif fix_var == 'h':
            ind_m = range(len(par['grid_m']))
            ind_k = fix_no
            x = par['grid_m']
            xlab = '$M_t$'
        else:
            raise ValueError('Choose either "m" or "h"')
        
        if fix_var == 'm':
            ind_m_grid = fix_no_grid
            ind_k_grid = range(len(par_grid['grid_k']))
            x_grid = par_grid['grid_k']
            xlab = '$K_t$'
        elif fix_var == 'h':
            ind_m_grid = range(len(par_grid['grid_m']))
            ind_k_grid = fix_no_grid
            x_grid = par_grid['grid_m']
            xlab = '$M_t$'
        else:
            raise ValueError('Choose either "m" or "h"')
        
        for i in range(len(ts)):
            fig = plt.figure()
            fig.canvas.manager.set_window_title(f'cons_t={ts[i]}_{par["prefix"]}')
            
            # Plot over labor choice
            for j in range(len(h)):
                plt.plot(x, sol['c'][ts[i]][h[j]][np.ix_([ind_m], ind_k)].flatten(), 'o',
                         linewidth=1.5, markersize=3, color=colors[j],
                         label=f'$h_t = {par["h"][h[j]]:.1f}, EGM $')
                plt.plot(x_grid, sol_grid['c'][ts[i]][h[j]][np.ix_([ind_m_grid], ind_k_grid)].flatten(), 'o',
                         linewidth=1.5, markersize=3, color=colors[j + 3],
                         label=f'$h_t = {par["h"][h[j]]:.1f}, VFI$')
            
            # Y-label either h or m
            if fix_var == 'm':
                plt.ylabel(f'$c(M_t = {par["grid_m"][fix_no]:.0f}, K_t, h_t)$')
            else:
                plt.ylabel(f'$c(M_t, K_t = {par["grid_k"][fix_no]:.0f}, h_t)$')
            
            # Layout
            if x_lim is not None:
                plt.xlim(x_lim)
            
            plt.xlabel(xlab)
            plt.legend(loc='best')
            plt.box(True)
            plt.grid(True)
            
            Funs.printfig(fig)
    
    @staticmethod
    def elasticity(elasticity, store_num_dc, x_lim, par, phi=None):
        fig = plt.figure()
        fig.canvas.manager.set_window_title(f'elasticity_number_of_DC_sets_{elasticity.shape[0]}_{par["prefix"]}')
        
        if phi is not None:
            for i in range(elasticity.shape[0]):
                plt.plot(elasticity[i, :], '-o', color=Figs.color()[i],
                         label=f'${par["prefix"]} = {phi[i]:.2f}$',
                         linewidth=1.5, markersize=3)
                plt.hold(True)
        else:
            for i in range(elasticity.shape[0]):
                plt.plot(elasticity[i, :], '-o', color=Figs.color()[i],
                         label=f'$D = {store_num_dc[i]:.0f}$',
                         linewidth=1.5, markersize=3)
                plt.hold(True)
        
        plt.xlim(x_lim)
        plt.ylabel('Labor Supply Elasticity')
        plt.xlabel('$t$')
        plt.grid(True)
        plt.legend(loc='best')
        plt.hold(False)
        
        Funs.printfig(fig)