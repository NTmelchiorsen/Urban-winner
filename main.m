%% *DC Choice Model with Human Capital Accumulation*
clear;
clc;
close all;
seed = 2017;
rng(seed);
funs.layout();

% source in mex file
setenv('MW_MINGW64_LOC','C:\TDM-GCC-64')
mex -setup C++
mex upper_envelope.cpp
clc;

%% 1: EGM - preferred model
%% 1.a. Solve the model with EGM
par = model_setup.setup();
par = model_setup.create_grids(par);
par.prefix = '';
sol = model.solve(par);

%% 1.b. Simulate 
sim = struct();
sim.N = 10^5;
sim.m_ini = 1.5;
sim.k_ini = 1;
sim = model.simulate(sim, sol, par, seed);
euler = sim.log10_euler_error_using_nan

%% 1.c. Figures - Presentation of solution
par.prefix = 'solution_pres';
ts_mid = 20;
ts_high = 35;
h_choice = 1:par.Nh;

figs.value_1d_fig(par, sol, ts_mid, h_choice, 'm', 77);
figs.value_1d_fig(par, sol, ts_high, h_choice, 'm', 77);

%Labour choices
figs.sim_choice_fig(par, sim);

%Mean of labour hours
figs.sim_mean_hhours(sim, par);

%h over life
par.prefix = 'solution_pres_capital';
figs.sim_mean_fig_new(sim.k, par);

%m over life
par.prefix = 'solution_pres_wealth';
figs.sim_mean_fig_new(sim.m, par);

%% 2: Compare EGM with grid search
%% 2.a. Solve the model with EGM (T=10)
par = model_setup.setup();
%Modify setup:
par.T = 10;
par.prefix = '';
par = model_setup.create_grids(par);
sol = model.solve(par);

%% 2.b. Solve with grid search (T=10 and small grid)
par_grid = model_setup.setup();
% Modify setup
par_grid.T = 10;
par_grid.Nc = 100;
par_grid.Nm = 50;
par_grid.Na = 50;
par_grid.Nk = 50;
par_grid = model_setup.create_grids(par_grid);
[sol_grid,par_grid] = model.solve_gridsearch(par_grid);

%% 2.c. Simulate - EGM
sim = struct();
sim.N = 10^5;
sim.m_ini = 1.5;
sim.k_ini = 1;
sim = model.simulate(sim, sol, par, seed);
euler = sim.log10_euler_error_using_nan

%% 2.d. Simulate - grid search
sim_grid = struct();
sim_grid.N = 10^5;
sim_grid.m_ini = 1.5;
sim_grid.k_ini = 1;
sim_grid = model.simulate(sim_grid, sol_grid, par_grid, seed);
euler_grid = sim_grid.log10_euler_error_using_nan

%% 2.e. Figures - Compare EGM and grid search solutions
%Value functions
ts_mid = 5;
ts_high = 8;
h_choice = 1:par.Nh;
par.prefix = 'compare_f_capital';

%Value functions (as func of K)
figs.value_1d_fig_grid_compare(par, par_grid, sol, sol_grid, ts_mid, h_choice, 'm', 77, 18, [0, 5]);
figs.value_1d_fig_grid_compare(par, par_grid, sol, sol_grid, ts_high, h_choice, 'm', 77, 18, [0, 5]);

par.prefix = 'compare_f_wealth';
%Value functions (as func of M)
figs.value_1d_fig_grid_compare(par, par_grid, sol, sol_grid, ts_mid, h_choice, 'h', 31, 11, [0, 10]);
figs.value_1d_fig_grid_compare(par, par_grid, sol, sol_grid, ts_high, h_choice, 'h', 31, 11, [0, 10]);

%Consumption functions (as func of M)
figs.cons_1d_fig_grid_compare(par, par_grid, sol, sol_grid, ts_mid, h_choice, 'h', 31, 11, [0, 10]);
figs.cons_1d_fig_grid_compare(par, par_grid, sol, sol_grid, ts_high, h_choice, 'h', 31, 11, [0, 10]);

%% 3. Sensitivity to parameters in human capital accumulation
par = model_setup.setup();
par.prefix = '';
par = model_setup.create_grids(par);

N = 10^5;
m_ini = 1.5;
k_ini = 1;

%% 3.a. phi_1
par_name = 'phi_1';
par_grid = {0.15, 0.2, 0.25};

% store solutions and simulations
store = model.sol_dif_pars(par, par_name, par_grid, N, m_ini, k_ini, seed);

% plot simulations
vars = {'hours', 'capital'};
figs.sim_mean_dif_pars(store, vars);

%% 3.b. phi_2
par_name = 'phi_2';
par_grid = {0.45, 0.6, 0.75};

% store solutions and simulations
store = model.sol_dif_pars(par, par_name, par_grid, N, m_ini, k_ini, seed);

% plot simulations
vars = {'hours', 'capital'};
figs.sim_mean_dif_pars(store, vars);

%% 4. Vary number of discrete labour choices
par = model_setup.setup();
par = model_setup.create_grids(par);

%Solve - 3 labour choices
sol = model.solve(par);
sim = struct();
sim.N = 10^5;
sim.m_ini = 1.5;
sim.k_ini = 1;
sim = model.simulate(sim, sol, par, seed);

%Solve - 5 labour choices 
par.h = [0; 0.25; 0.5; 0.75; 1];
par.Nh = numel(par.h);
sol = model.solve(par);
sim_five = sim;
sim_five = model.simulate(sim_five, sol, par, seed);
par.prefix = 'D=5';
figs.sim_choice_fig(par, sim_five);
figs.sim_mean_hhours_vary_D(sim, sim_five.means.hours, 5, par);

%Solve - 9 labour choices
par.h = [0; 0.125; 0.25; 0.375; 0.5; 0.625; 0.75; 0.875; 1];
par.Nh = numel(par.h);
sol = model.solve(par);
sim_nine = sim;
sim_nine = model.simulate(sim_nine, sol, par, seed);
par.prefix = 'D=9';
figs.sim_choice_fig(par, sim_nine);
figs.sim_mean_hhours_vary_D(sim, sim_nine.means.hours, 9, par);

%Solve - 7 labour choices
par.h = [0; 0.25; 0.5; 0.75; 1; 1.25; 1.5];
par.Nh = numel(par.h);
sol = model.solve(par);
sim_seven = sim;
sim_seven = model.simulate(sim_seven, sol, par, seed);
par.prefix = 'D=7';
figs.sim_choice_fig(par, sim_seven);
figs.sim_mean_hhours_vary_D(sim, sim_seven.means.hours, 7, par);

%% 5. Permanent tax change
%% 5.a. Unexpected permanent tax at time t=5
par = model_setup.setup();
par.prefix = 'base';
par = model_setup.create_grids(par);
par.tax_rate = 0.35; % pre tax rate
post_tax = 0.30;
time = 5; %tax cut period
tax = model.perm_tax_time(par.tax_rate, post_tax, par, seed, time);
elasticity = model.reaction(tax);
figs.elasticity(elasticity, par.Nh, [time, par.T], par);

%% 5.b. Unexpected permanent tax at time t=5 - exogenous human capital
par = model_setup.setup();
par.prefix = 'exog_human_cap';
par = model_setup.create_grids(par);
par.delta=-0.02; %exogenous growth in human capital
par.phi_1=0; %no endogenous accumulation
par.tax_rate = 0.35; % pre tax rate
post_tax = 0.30;
time = 5; %tax cut period
tax = model.perm_tax_time(par.tax_rate, post_tax, par, seed, time);
elasticity = model.reaction(tax);
figs.elasticity(elasticity, par.Nh, [time, par.T], par);

%% 5.c. Unexpected permanent tax at time t=10, t=20, t=30
par = model_setup.setup();
par = model_setup.create_grids(par);
par.tax_rate = 0.35; % pre tax rate
post_tax = 0.30;
for t = [10,20,30]
    par.prefix = sprintf('taxtime_%d', t);
    time = t; %tax cut period
    tax = model.perm_tax_time(par.tax_rate, post_tax, par, seed, time);
    elasticity = model.reaction(tax);
    figs.elasticity(elasticity, par.Nh, [time, par.T], par);
end

%% 5.d. Tax - sensitivity to labour choices
time = 5;
discrete_choices = cell(3,1);
discrete_choices{1,1} = [0;0.5;1];
discrete_choices{2,1}= [0;0.25;0.5;0.75;1];
discrete_choices{3,1}= [0;0.125;0.25;0.375;0.5;0.625;0.75;0.875;1];
store_num_dc = nan(3,1);
elasticity = nan(3,par.T-1);

for i = [1:numel(discrete_choices)]
    par.h = discrete_choices{i,1};
    par.Nh = numel(par.h);
    tax = model.perm_tax_time(par.tax_rate, post_tax, par, seed, time);
    elasticity(i,:) = model.reaction(tax);
    store_num_dc(i,1) = par.Nh;
end

par.prefix = sprintf('taxtime_%d', time);
figs.elasticity(elasticity,store_num_dc, [time, par.T], par);

%% 5.c. Tax - sensitivity to human capital accumulation parameters
par = model_setup.setup();
par.prefix = '-';
par = model_setup.create_grids(par);
par.tax_rate = 0.35; % pre tax rate
post_tax = 0.30;
time = 5; %tax cut period

% Phi_1
phi_1 = {0.15,0.2,0.25};
elasticity = nan(3,par.T-1);
store_num_dc = nan(3,1);

for i = [1:numel(phi_1)]
   par.phi_1 = phi_1{i};
   tax = model.perm_tax_time(par.tax_rate, post_tax, par, seed, time);
   elasticity(i,:) = model.reaction(tax);
   store_num_dc(i,1) = 3;
end

par.prefix = 'phi_1';
figs.elasticity(elasticity, store_num_dc, [time, par.T], par, phi_1);

