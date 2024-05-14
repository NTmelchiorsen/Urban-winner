classdef model_setup
methods(Static)

    function par = setup()
        
        par = struct();
        
        % 1. demograhpics
        par.T = 40;
        
        % 2. preferences
        par.rho = 0.75;
        par.beta = 0.95;
        par.alpha = 1.5;
        par.lw = 1.8;
        
        % 3. labour choices and human capital accumulation
        par.h = [0; 0.5; 1];
        par.Nh = numel(par.h);
        par.delta = 0.1;
        par.phi_1 = 0.2;
        par.phi_2 = 0.6;
        
        % 4. taste shocks
        par.sigma_eta = 0.25; % taste shocks
        
        % 4. income 
        par.tax_rate = 0.0;
        par.kappa = 1;
        par.sigma_xi = 0.1; % capital uncertainty
        par.UB = 0.0;
        
        % 5. saving
        par.R = 1.05;
        
        % 6. grids and numerical integration
        par.m_max = 20.0;
        par.m_phi = 1.1; % curvature parameters
        par.m_points_low = 50;
        par.a_max = 20.0;
        par.a_phi = 1.1; % curvature parameters
        par.k_max = 10.0;
        par.k_phi = 1.0; % curvature parameters
        
            % number of elements
            par.Nxi  = 8;
            par.Nm = 200;            
            par.Na = 200;
            par.Nk = 150;
    end
    function par = create_grids(par)
        
        % 1. check parameters
        assert(par.rho >= 0, 'not rho > 0');
        assert(par.Nm == par.Na, 'Nm should equal Na');        
                
        % 2. shocks        
        [par.xi, par.xi_w] = funs.GaussHermite_lognorm(par.sigma_xi,par.Nxi);
                     
        % 3. end-of-period assets
        par.grid_a = cell(par.T,1);
        for t = 1:par.T
            par.grid_a{t} = funs.nonlinspace(0+1e-6,par.a_max,par.Na,par.a_phi);
        end     
        
        % 4. cash-on-hand
        par.grid_m = funs.nonlinspace(1e-4, par.m_max, par.Nm, par.m_phi);
        
        % 5 human capital
        par.grid_k = funs.nonlinspace(1e-4, par.k_max, par.Nk, par.k_phi);
    end
end
end