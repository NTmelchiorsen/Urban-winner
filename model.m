classdef model
methods(Static)
    
    %%%% Utility function %%%%
    function u = utility(c, h_choice, par) %før l_choice
        h_hour = par.h(h_choice);
        u = c.^(1-par.rho)/(1-par.rho) - par.lw*h_hour.^par.alpha/par.alpha;
    end    
    function u = marg_utility_c(c, par)
        u = c.^(-par.rho);            
    end
    function c = inv_marg_utility_c(u, par)
        c = u.^(-1/par.rho);            
    end
    
    %%%% Transitions %%%%
    function [m_plus] = m_trans(a, k, k_plus, h_choice, par)   
        h_hour = par.h(h_choice);
        k = repmat(k, 1, size(k_plus,2));
        m_plus = par.R .* a + par.kappa * (1-par.tax_rate) * h_hour .* k + (h_hour == 0) * par.UB;
    end
    function [k_plus] = k_trans(k, h_choice, xi, par)
        h_hour = par.h(h_choice);
        k_plus = ((1 - par.delta) * k + par.phi_1 * h_hour .^ par.phi_2) .* xi; % xi is the shock
    end
    
    %%%% Expectations %%%%
    function [V_plus, prob] = taste_exp(m_plus, k_plus, v_plus_interp, par)
        
        % 1. v_plus (choice-specific value functions)  
        v_matrix = zeros(numel(m_plus), par.Nh);
        for i_nh = 1:par.Nh % loop over labor choices                         
            v_matrix(:, i_nh) = reshape(v_plus_interp{i_nh}([m_plus(:) k_plus(:)]), size(m_plus(:))); % stack them next to each other
        end
        
        % 2. find logsum and choice probabilities
        [V_plus, prob] = funs.logsum_vec(v_matrix, par.sigma_eta);  
        
        % put back into shape
        V_plus = reshape(V_plus, size(m_plus)); 
        
        % dimensions
        % V_plus is Nm -by- Nxi (#grid points in m -by- #gauss-hermite points)
        % prob is Nm*Nxi -by- NL (#grid points in m * #gauss-hermite points -by- #labor choices) 
    end  
    function [avg_marg_u_plus] = future_marg_u(c_plus_interp, m_plus, k_plus, prob, w, par)
        
        % 1. next-period consumption
        c_plus = cell(par.Nh, 1);  
        for i_nh = 1:par.Nh % loop over labor choices                         
            c_plus{i_nh} = reshape(c_plus_interp{i_nh}([m_plus(:) k_plus(:)]), size(m_plus(:)));      
        end

        % 2. expected future marginal utility
        marg_u_plus_matrix = zeros(numel(m_plus), par.Nh);
        for i_nh = 1:par.Nh % loop over labor choices                         
            marg_u_plus_matrix(:, i_nh) = model.marg_utility_c(c_plus{i_nh}, par); % stack them next to each other
        end      
                                
        marg_u_plus_taste = sum(prob .* marg_u_plus_matrix, 2); % expectation over taste shocks
        marg_u_plus_taste = reshape(marg_u_plus_taste, size(m_plus)); % put back into shape so dim is Nm -by- Nxi
        avg_marg_u_plus = sum(w .* marg_u_plus_taste, 2); % expectation over human capital shocks so dim is Nm -by- 1
    end
    function [V] = value_of_choice_gridsearch(C,h,mt,kt,last,par)
        
        if last == 1 % last period
            V = model.utility(C,h,par);
        else
            K_plus = ((1-par.delta)*kt+par.phi_1*par.h(h)^par.phi_2).*par.xi;
            kt = repmat(kt, 1, size(K_plus,1));
            M_plus = par.R*(mt-C)+par.kappa*par.h(h).*kt +(par.h(h)==0)*par.UB;
            V1 = zeros(numel(M_plus), 1);
            V2 = zeros(numel(M_plus), 1);
            V3 = zeros(numel(M_plus), 1);
            V1(:) = par.v_plus_interp{1}([M_plus(:) K_plus(:)]);
            V2(:) = par.v_plus_interp{2}([M_plus(:) K_plus(:)]);
            V3(:) = par.v_plus_interp{3}([M_plus(:) K_plus(:)]);
            V  = model.utility(C,h,par) + sum(par.xi_w.*par.beta.*funs.logsum_2(V1,V2,V3,par.sigma_eta));
        end
    end
    
    %%%% Finding optimal consumption %%%%    
    function [c_raw, m_raw, v_plus_raw] = egm(t, h_choice, k, v_plus_interp, c_plus_interp, par)

        % 1. assets, human capit        
        a = par.grid_a{t};   
        k = repmat(k, [par.Na 1]);     
        w = repmat(par.xi_w', [par.Na 1]); %Na-by-Nxi
        xi = repmat(par.xi', [par.Na 1]); %Na-by-Nxi
        
        % 2. next-period ressources and value
        k_plus = model.k_trans(k, h_choice, xi, par);                
        m_plus = model.m_trans(a, k, k_plus, h_choice, par);
        [v_plus_vec_raw, prob] = model.taste_exp(m_plus, k_plus, v_plus_interp, par); % expectation over taste shocks
        v_plus_raw = sum(w .* v_plus_vec_raw, 2); %expectation over human capital shocks
        
        % 3. Expected future marginal utility
        avg_marg_u_plus = model.future_marg_u(c_plus_interp, m_plus, k_plus, prob, w, par);       
        
        % 4. raw c, m, and v (output from EGM - but possibility of correspondences instead of functions)
        c_raw = model.inv_marg_utility_c(par.beta * par.R * avg_marg_u_plus, par);        
        m_raw = par.grid_a{t} + c_raw;               
    end
    function [c, v] = upper_envelope_cpp(c_raw, m_raw, v_plus_raw, h_choice, t, par)
        
        % 1. add point at bottom
        c_raw = [1e-6; c_raw];
        m_raw = [1e-6; m_raw];        
        a_raw = [0; par.grid_a{t}];
        v_plus_raw = [v_plus_raw(1); v_plus_raw];
        
        % 2. call the function
        h_hour = par.h(h_choice);
        [c, v] = upper_envelope(c_raw, m_raw, v_plus_raw, a_raw, par.grid_m, h_hour, par.rho, par.alpha, par.beta, par.lw);        
    end
    function [c, v] = upper_envelope(c_raw, m_raw, v_plus_raw, h_choice, t, par)

        % 1. add point at bottom
        c_raw = [1e-6; c_raw];
        m_raw = [1e-6; m_raw];        
        a_raw = [0; par.grid_a{t}];
        v_plus_raw = [v_plus_raw(1); v_plus_raw]; 
        
        % 2. preallocate
        c = nan(par.Nm, 1);
        v = -inf*ones(par.Nm, 1);        
        
        % 3. upper envelope (select optimal consumption)     
        for i = 1:numel(m_raw)-1
            
            % a. current intervals and slopes
            m_low   = m_raw(i);
            m_high  = m_raw(i+1);
            c_slope = (c_raw(i+1) - c_raw(i))/(m_high - m_low);     
            
            a_low = a_raw(i);
            a_high = a_raw(i+1);
            v_plus_slope = (v_plus_raw(i+1) - v_plus_raw(i)) / (a_high - a_low);
            
            % b. loop through common grid
            for j = 1:par.Nm              
                
                m_now = par.grid_m(j);
                do_interp = m_now >= m_low && m_now <= m_high;
                do_extrap = m_now > m_high && i == numel(m_raw)-1;
                if do_interp || do_extrap                    
                
                    % i. consumption
                    c_guess = c_raw(i) + c_slope*(m_now - m_low);

                    % ii. post-decision value
                    a_guess = m_now - c_guess;
                    v_plus = v_plus_raw(i) + v_plus_slope * (a_guess - a_low);
                    
                    % iii. value-of-choice
                    v_guess = model.utility(c_guess, h_choice, par) + par.beta * v_plus;
                
                    % iii. update
                    if v_guess > v(j)
                        v(j) = v_guess;
                        c(j) = c_guess;
                    end
                    
                end % if 
            end % common grid
        end % i loop               
    end
    function [V, Cstar] = gridsearch(par,h,last)
        
        % loop over states    
        V     = zeros(par.Nm,par.Nk);
        Cstar = nan(par.Nm,par.Nk);   
        par.grid_C = linspace(0,1,par.Nc);
        for i_M = 1:numel(par.grid_m)   
            mt = par.grid_m(i_M);
            
            for i_K = 1:numel(par.grid_k)
            kt = par.grid_k(i_K);

                for ic = 1:numel(par.grid_C)               
                    C = par.grid_C(ic)*mt;

                    % a. find value of choice       
                    V_new = model.value_of_choice_gridsearch(C,h,mt,kt,last,par);

                   % b. save if V_new>V
                    if V_new>V(i_M,i_K)
                        V(i_M,i_K)     = V_new;
                        Cstar(i_M,i_K) = C;
                    end
                end
            end
        end     
    end
    
    %%%% Solution %%%%
    function sol = solve(par)
        
        % 1. allocate solution struct
        sol = struct();
        sol.c = cell(par.T, par.Nh);
        sol.v = cell(par.T, par.Nh);
        sol.v_plus = cell(par.T, par.Nh);        
        
        % 2. last period (=consume all)          
        for i_nh = 1:par.Nh            
            sol.m{par.T, i_nh} = repmat(par.grid_m, [1, par.Nk]);
            sol.c{par.T, i_nh} = repmat(par.grid_m, [1, par.Nk]);
            sol.v{par.T, i_nh} = model.utility(sol.c{par.T, i_nh}, i_nh, par);
        end
        
        % 3. before last period
        c_plus_interp = cell(par.Nh, 1);
        for t = par.T-1:-1:1
            t
            % a. preallocate
            for i_nh = 1:par.Nh
               sol.c{t, i_nh} = nan(par.Nm, par.Nk);
               sol.v{t, i_nh} = nan(par.Nm, par.Nk);               
            end
            
            % b. create interpolants                                        
            for i_nh = 1:par.Nh % loop over labor choices                         
                c_plus_interp{i_nh} = griddedInterpolant({par.grid_m, par.grid_k}, sol.c{t+1, i_nh}, 'linear');
                v_plus_interp{i_nh} = griddedInterpolant({par.grid_m, par.grid_k}, sol.v{t+1, i_nh}, 'linear'); 
            end
            
            % c. solve by EGM            
            for i_nh = 1:par.Nh % loop over labor choices
                for i_k = 1:par.Nk
                    
                    % unpack state
                    k = par.grid_k(i_k);
                    
                    % egm
                    [c_raw, m_raw, v_plus_raw] = model.egm(t, i_nh, k, v_plus_interp, c_plus_interp, par); % raw
                    %[c, v] = model.upper_envelope(c_raw, m_raw, v_plus_raw, i_nl, t, par); % upper envelope
                    [c, v] = model.upper_envelope_cpp(c_raw, m_raw, v_plus_raw, i_nh, t, par); % upper envelope                    
                    
                    % store solution
                    sol.c{t, i_nh}(:,i_k) = c;
                    sol.v{t, i_nh}(:,i_k) = v;                    
                end
            end  
            
        end % t loop                        
    end 
    function store = sol_dif_pars(par, par_name, par_grid, N, m_ini, k_ini, seed)
        
        % 1. initialize
            % i. par
            par.prefix = par_name;

            % ii. sim       
            sim = struct();
            sim.N = N;
            sim.m_ini = m_ini;
            sim.k_ini = k_ini;
            
            % iii. store
            store = struct();
            store.par_grid = par_grid;
            
        % 2. solve and simulate
        for i = 1:numel(par_grid)

            % par
            par.(par_name) = par_grid{i}; %overwrite parameter
            store.par{i} = par;
    
            % sol
            store.sol{i} = model.solve(store.par{i});
    
            % simulate
            store.sim{i} = model.simulate(sim, store.sol{i}, store.par{i}, seed);
        end
    end
    function store = sol_dif_pars_tax(par, par_name, par_grid, N, m_ini, k_ini, seed, time)
        
        % 1. initialize
            % i. par
            par.prefix = sprintf(par_name);

            % ii. sim       
            sim = struct();
            sim.N = N;
            sim.m_ini = m_ini;
            sim.k_ini = k_ini;
            
            % iii. store
            store = struct();
            store.par_grid = par_grid;
            
        % 2. solve and simulate
        for i = 1:numel(par_grid)

            % par
            par.(par_name) = par_grid{i}; %overwrite parameter
            store.par{i} = par;
    
            % sol
            store.sol{i} = model.solve(store.par{i});
            
        end
         % simulate
        store.sim{1} = model.simulate(sim, store.sol{1}, store.par{1}, seed);
        store.sim{2} = model.simulate_tax(sim, store.sol{1}, store.par{1}, store.sol{2}, store.par{2}, seed,time);
        
    end
    function [sol,par] = solve_gridsearch(par)
        
        % 1. C-grid, M-grid and H-grid
        par.grid_C = linspace(0,1,par.Nc);
       
        % 2. Allocate memory
        sol = struct();
        
        sol.c = cell(par.T, par.Nh);
        sol.v = cell(par.T, par.Nh);
        
        par.v_plus_interp = cell(par.Nh, 1);
        
        % 3. Last period
        for h = 1:par.Nh
            [sol.v{par.T,h},sol.c{par.T,h}] = model.gridsearch(par,h,1);
        end
        
        % 3. backwards over time
        for t = par.T-1:-1:1
            t
            % a. interpolant
            for h = 1:par.Nh
                par.v_plus_interp{h} = griddedInterpolant({par.grid_m, par.grid_k},sol.v{t+1,h},'linear');   
            end

            % b. find V for all discrete choices and states
            for h = 1:par.Nh
                [sol.v{t,h},sol.c{t,h}] = model.gridsearch(par,h,0);
            end

        end
        
    end 
    
    %%%% Simulation %%%%
    function sim = euler_error(sim)
        abs_error = abs(sim.lhs_euler - sim.rhs_euler);
        
        abs_error_0 = abs_error + (abs_error == 0); % set zeros equal to 1
        log10_abs_error_0 = log10(abs_error_0 ./ sim.c(:, 1:(end-1))); %this mean includes all the zeros (log(1) = 0)
        
        abs_error_nan = abs_error; abs_error_nan(abs_error == 0) = nan; % set zeros equal to nan
        log10_abs_error_nan = log10(abs_error_nan ./ sim.c(:, 1:(end-1))); %this mean excludes all the zeros
        
        sim.euler_error = nanmean(nanmean(abs_error));
        sim.log10_euler_error = nanmean(nanmean(log10_abs_error_0));
        sim.log10_euler_error_using_nan = nanmean(nanmean(log10_abs_error_nan));
    end
    function sim = simulate(sim, sol, par, seed)
    
        % 1. set seed and initialize
        rng(seed);
        sim.m = sim.m_ini * ones(sim.N, par.T);
        sim.k = sim.k_ini * ones(sim.N, par.T);
        
        % 2. allocate
        sim.c = nan(sim.N, par.T);
        sim.h_choice = nan(sim.N, par.T);
        sim.a = zeros(sim.N, par.T);        
        sim.lhs_euler = nan(sim.N, par.T-1);
        sim.rhs_euler = nan(sim.N, par.T-1);
        
        % 3. random draws
        unif = rand(sim.N, par.T);
        shock = exp(randn(sim.N, par.T-1) * par.sigma_xi);
        
        % 4. simulate
        for t = 1:par.T
            
            % a. values of discrete choices and interpolants
            v_matrix = nan(size(sim.m(:,t), 1), par.Nh);
            c_interp = cell(par.Nh, 1);
            c_plus_interp = cell(par.Nh, 1);
            
            for i_nh = 1:par.Nh % loop over labour choices                
                v_interp = griddedInterpolant({par.grid_m, par.grid_k}, sol.v{t, i_nh}, 'linear');
                v_matrix(:, i_nh) = reshape(v_interp([sim.m(:,t) sim.k(:,t)]), size(sim.m(:,t)));  
                c_interp{i_nh} = griddedInterpolant({par.grid_m, par.grid_k}, sol.c{t, i_nh}, 'linear');
                if t < par.T
                    c_plus_interp{i_nh} = griddedInterpolant({par.grid_m, par.grid_k}, sol.c{t+1, i_nh}, 'linear');                
                end
            end
            
            % b. choice-specific probabilities
            [~, prob] = funs.logsum_vec(v_matrix, par.sigma_eta);
            
            % c. actual labour choice            
            prob_cum = cumsum(prob, 2); %cumsum across cols, so last col equals 1
            I = sum(unif(:,t) > prob_cum, 2) + 1; %find the first col of prob_cum that exceeds unif. This is the right choice            
            sim.h_choice(:, t) = I; %1 is no labour, last is full labour                        
            
            % d. actual consumption choice
            for i_nh = 1:par.Nh % loop over labour choices
               ind = (I == i_nh); % find indices, where the agent choise l_choice = i
               sim.c(ind, t) = c_interp{i_nh}([sim.m(ind, t) sim.k(ind, t)]); % extract right consumption 
            end

            % e. next period
            if t < par.T % in last period there are no interior solutions (everything is consumed)                                
                     
                % 1. update states                
                % i. assets
                sim.a(:,t) = sim.m(:,t) - sim.c(:,t);            
                
                % ii. pull out
                a = sim.a(:,t);
                k = sim.k(:,t);
                h_choice = sim.h_choice(:,t);
                xi = shock(:,t);
                
                % iii. next period ressources and human capital                
                k_plus = model.k_trans(k, h_choice, xi, par);
                
                %par.Nxi=1; %to get right m dimension
                m_plus = model.m_trans(a, k, k_plus, h_choice, par);
                
                % iv. remember to update
                sim.k(:,t+1) = k_plus;
                sim.m(:,t+1) = m_plus;               

                % 2. expected future marginal utility
                avg_marg_u_plus = model.future_marg_u(c_plus_interp, m_plus, k_plus, prob, 1, par);
                
                % 3. lhs. and rhs. of euler
                sim.lhs_euler(:,t) = model.marg_utility_c(sim.c(:,t), par); %present consumption was found earlier
                sim.rhs_euler(:,t) = par.beta * par.R * avg_marg_u_plus;
                
                % 4. ignore corner solutions
                corner_ind = (a < 1e-5); % if c = m
                sim.lhs_euler(corner_ind, t) = nan;
                sim.rhs_euler(corner_ind, t) = nan;                
            end
        end
        
        % 5. calculate moments/statistics
        % i. labor supply
            % 1. labor choice frequencies
            sim.labor = nan(par.Nh, par.T);
            for i = 1:par.Nh % loop over labour choices
                sim.labor(i, :) = sum((sim.h_choice == i)) / sim.N;        
            end
            
            % 2. average hours
            sim.means.hours = sum(par.h .* sim.labor);
            
        % ii. consumption
        sim.means.cons = mean(sim.c);
        
        % iii. assets
        sim.means.assets = mean(sim.a);
        sim.mean_a = mean(sim.a); 
        
        % iv. cash on hand
        sim.means.cash = mean(sim.m);        
        
        % v. human capital
        sim.means.capital = mean(sim.k);

        % vi. wages
        sim.means.wage = mean(par.kappa * sim.k);
         
        % vii. euler errors
        sim = model.euler_error(sim);        
    end
    function sim = simulate_tax(sim, sol_old, par_old, sol_tax, par_tax, seed,time)
    
        % 1. set seed and initialize
        rng(seed);
        par = par_old;
        sim.m = sim.m_ini * ones(sim.N, par.T);
        sim.k = sim.k_ini * ones(sim.N, par.T);
        
        % 2. allocate
        sim.c = nan(sim.N, par.T);
        sim.h_choice = nan(sim.N, par.T);
        sim.a = zeros(sim.N, par.T);        
        sim.lhs_euler = nan(sim.N, par.T-1);
        sim.rhs_euler = nan(sim.N, par.T-1);
        
        % 3. random draws
        unif = rand(sim.N, par.T);
        shock = exp(randn(sim.N, par.T-1) * par.sigma_xi);
        
        % 4. simulate
        for t = 1:time-1
            sol = sol_old;
            par = par_old;
            % a. values of discrete choices and interpolants
            v_matrix = nan(size(sim.m(:,t), 1), par.Nh);
            c_interp = cell(par.Nh, 1);
            c_plus_interp = cell(par.Nh, 1);
            
            for i_nh = 1:par.Nh % loop over labour choices                
                v_interp = griddedInterpolant({par.grid_m, par.grid_k}, sol.v{t, i_nh}, 'linear');
                v_matrix(:, i_nh) = reshape(v_interp([sim.m(:,t) sim.k(:,t)]), size(sim.m(:,t)));  
                c_interp{i_nh} = griddedInterpolant({par.grid_m, par.grid_k}, sol.c{t, i_nh}, 'linear');
                if t < par.T
                    c_plus_interp{i_nh} = griddedInterpolant({par.grid_m, par.grid_k}, sol.c{t+1, i_nh}, 'linear');                
                end
            end
            
            % b. choice-specific probabilities
            [~, prob] = funs.logsum_vec(v_matrix, par.sigma_eta);
            
            % c. actual labour choice            
            prob_cum = cumsum(prob, 2); %cumsum across cols, so last col equals 1
            I = sum(unif(:,t) > prob_cum, 2) + 1; %find the first col of prob_cum that exceeds unif. This is the right choice            
            sim.h_choice(:, t) = I; %1 is no labour, last is full labour                        
            
            % d. actual consumption choice
            for i_nh = 1:par.Nh % loop over labour choices
               ind = (I == i_nh); % find indices, where the agent choise l_choice = i
               sim.c(ind, t) = c_interp{i_nh}([sim.m(ind, t) sim.k(ind, t)]); % extract right consumption 
            end

            % e. next period
            if t < par.T % in last period there are no interior solutions (everything is consumed)                                
                     
                % 1. update states                
                % i. assets
                sim.a(:,t) = sim.m(:,t) - sim.c(:,t);            
                
                % ii. pull out
                a = sim.a(:,t);
                k = sim.k(:,t);
                h_choice = sim.h_choice(:,t);
                xi = shock(:,t);
                
                % iii. next period ressources and human capital                
                k_plus = model.k_trans(k, h_choice, xi, par);
                m_plus = model.m_trans(a, k, k_plus, h_choice, par);
                
                % iv. remember to update
                sim.k(:,t+1) = k_plus;
                sim.m(:,t+1) = m_plus;               

                % 2. expected future marginal utility
                avg_marg_u_plus = model.future_marg_u(c_plus_interp, m_plus, k_plus, prob, 1, par);
                
                % 3. lhs. and rhs. of euler
                sim.lhs_euler(:,t) = model.marg_utility_c(sim.c(:,t), par); %present consumption was found earlier
                sim.rhs_euler(:,t) = par.beta * par.R * avg_marg_u_plus;
                
                % 4. ignore corner solutions
                corner_ind = (a < 1e-5); % if c = m
                sim.lhs_euler(corner_ind, t) = nan;
                sim.rhs_euler(corner_ind, t) = nan;                
            end
        end
        
        for t = time:par.T
            sol = sol_tax;
            par = par_tax;
            % a. values of discrete choices and interpolants
            v_matrix = nan(size(sim.m(:,t), 1), par.Nh);
            c_interp = cell(par.Nh, 1);
            c_plus_interp = cell(par.Nh, 1);
            
            for i_nh = 1:par.Nh % loop over labour choices                
                v_interp = griddedInterpolant({par.grid_m, par.grid_k}, sol.v{t, i_nh}, 'linear');
                v_matrix(:, i_nh) = reshape(v_interp([sim.m(:,t) sim.k(:,t)]), size(sim.m(:,t)));  
                c_interp{i_nh} = griddedInterpolant({par.grid_m, par.grid_k}, sol.c{t, i_nh}, 'linear');
                if t < par.T
                    c_plus_interp{i_nh} = griddedInterpolant({par.grid_m, par.grid_k}, sol.c{t+1, i_nh}, 'linear');                
                end
            end
            
            % b. choice-specific probabilities
            [~, prob] = funs.logsum_vec(v_matrix, par.sigma_eta);
            
            % c. actual labour choice            
            prob_cum = cumsum(prob, 2); %cumsum across cols, so last col equals 1
            I = sum(unif(:,t) > prob_cum, 2) + 1; %find the first col of prob_cum that exceeds unif. This is the right choice            
            sim.h_choice(:, t) = I; %1 is no labour, last is full labour                        
            
            % d. actual consumption choice
            for i_nh = 1:par.Nh % loop over labour choices
               ind = (I == i_nh); % find indices, where the agent choise l_choice = i
               sim.c(ind, t) = c_interp{i_nh}([sim.m(ind, t) sim.k(ind, t)]); % extract right consumption 
            end

            % e. next period
            if t < par.T % in last period there are no interior solutions (everything is consumed)                                
                     
                % 1. update states                
                % i. assets
                sim.a(:,t) = sim.m(:,t) - sim.c(:,t);            
                
                % ii. pull out
                a = sim.a(:,t);
                k = sim.k(:,t);
                h_choice = sim.h_choice(:,t);
                xi = shock(:,t);
                
                % iii. next period ressources and human capital                
                k_plus = model.k_trans(k, h_choice, xi, par);
                m_plus = model.m_trans(a, k, k_plus, h_choice, par);
                
                % iv. remember to update
                sim.k(:,t+1) = k_plus;
                sim.m(:,t+1) = m_plus;               

                % 2. expected future marginal utility
                avg_marg_u_plus = model.future_marg_u(c_plus_interp, m_plus, k_plus, prob, 1, par);
                
                % 3. lhs. and rhs. of euler
                sim.lhs_euler(:,t) = model.marg_utility_c(sim.c(:,t), par); %present consumption was found earlier
                sim.rhs_euler(:,t) = par.beta * par.R * avg_marg_u_plus;
                
                % 4. ignore corner solutions
                corner_ind = (a < 1e-5); % if c = m
                sim.lhs_euler(corner_ind, t) = nan;
                sim.rhs_euler(corner_ind, t) = nan;                
            end
        end
        
        % 5. calculate moments/statistics
        % i. labor supply
            % 1. labor choice frequencies
            sim.labor = nan(par.Nh, par.T);
            for i = 1:par.Nh % loop over labour choices
                sim.labor(i, :) = sum((sim.h_choice == i)) / sim.N;        
            end
            
            % 2. average hours
            sim.means.hours = sum(par.h .* sim.labor);
            
        % ii. consumption
        sim.means.cons = mean(sim.c);
        
        % iii. assets
        sim.means.assets = mean(sim.a);
        sim.mean_a = mean(sim.a); 
        
        % iv. cash on hand
        sim.means.cash = mean(sim.m);        
        
        % v. human capital
        sim.means.capital = mean(sim.k);

        % vi. wages
        sim.means.wage = mean(par.kappa * sim.k);
         
        % vii. euler errors
        sim = model.euler_error(sim);        
    end
    
    %%%% Tax experiments %%%%
    function elasticity = reaction(tax)
        after_tax = ((1 - tax.post_tax) / (1 - tax.pre_tax) - 1) * 100; %pct change in after tax rate
        pre_supply = mean(tax.pre_supply);
        post_supply = mean(tax.post_supply);        
        elasticity = (post_supply ./ pre_supply - 1) * 100 / after_tax;        
    end
 
    function tax = perm_tax_time(pre_tax, post_tax, par, seed, time)
        
        % set up tax struct
        tax = struct();
        tax.pre_tax = pre_tax;
        tax.post_tax = post_tax;
        
        % set up for solution and simulation
        par_name = 'tax_rate';
        par_grid = {pre_tax, post_tax};
        N = 10^5;
        m_ini = 1.5;
        k_ini = 1;
        
        % store solutions and simulations
        store = model.sol_dif_pars_tax(par, par_name, par_grid, N, m_ini, k_ini, seed, time);
        tax.pre_supply = par.h(store.sim{1}.h_choice(:, 1:(par.T - 1)));
        tax.post_supply = par.h(store.sim{2}.h_choice(:, 1:(par.T - 1)));        
    end
        
end % methods (static)
end %classdef model