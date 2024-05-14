classdef figs
methods(Static)
    
    function [color] = color()
        color = {'k','b','r','g','y','c','m',[102/255 0/255 51/255],[153/255 1 0]};   
    end
    
    % Simulation figures
    function [] = sim_choice_fig(par, sim)
        
        fig = figure('Name',sprintf('labor_choice_%s', par.prefix));

        color = figs.color();       
        for i = 1:par.Nh
            h = par.h(i);
            plot(1:par.T, sim.labor(i, :), '-o',...
                'linewidth', 1.5, 'MarkerSize', 3, 'color', color{i},...
                'DisplayName', sprintf('$h_t = %0.2f$', h))
            hold on;
        end                           
        
        % layout
        xlabel('$t$')
        legend('Location','best');
        box('on');
        grid on;
        
        funs.printfig(fig);        
    end
    function [] = sim_mean_fig_new(var, par)
        
        fig = figure('Name',sprintf('State_variable_%s', par.prefix));
        
        color = figs.color();
        
        plot(mean(var),'-o', 'linewidth', 1.5, 'MarkerSize', 3, 'color', color{1},'DisplayName', 'Mean');
        hold on;
        plot(prctile(var,2.5),'-o', 'linewidth', 1.5, 'MarkerSize', 3, 'color', color{2}, 'DisplayName', '2.5 prctile');
        plot(prctile(var,97.5),'-o', 'linewidth', 1.5, 'MarkerSize', 3, 'color', color{3}, 'DisplayName', '97.5 prctile');
        hold off;
        
        % layout
        xlabel('$t$')
        legend('Location','best');
        grid on;
        
        funs.printfig(fig);
    end
    function [] = sim_mean_fig(par, sim, vars)
                
        color = figs.color(); 
        
        for i = 1:numel(vars)            
            fig = figure('Name',sprintf('mean_%s_%s', [vars{i}{:}], par.prefix));
            for j = 1:numel(vars{i})  
                plot(1:par.T, sim.means.(vars{i}{j}), '-o',...
                     'linewidth', 1.5, 'MarkerSize', 3, 'color', color{j},...
                     'DisplayName', sprintf('$%s$', vars{i}{j}))
                hold on;
            end           
        
            % layout
            xlabel('$t$')
            legend('Location','best');
            box('on');
            grid on;            
                    
            funs.printfig(fig);            
        end
    end
    function [] = sim_mean_dif_pars(store, vars)
                
        color = figs.color(); 
        
        for i = 1:numel(vars)        
            fig = figure('Name',sprintf('mean_%s_%s', [vars{i}], store.par{1}.prefix));
            for j = 1:numel(store.par_grid)
                
                if numel(store.par_grid{j}) > 1
                    store.par_grid{j} = numel(store.par_grid{j});
                end
                
                plot(1:store.par{1}.T, store.sim{j}.means.(vars{i}), '-o',...
                     'linewidth', 1.5, 'MarkerSize', 3, 'color', color{j},...
                     'DisplayName', sprintf('$%s=%0.2f$', store.par{1}.prefix, store.par_grid{j}))
                hold on;
            end           
        
            % layout
            xlabel('$t$')
            
            if i==1
                ylabel('$Mean(h_t)$')
            elseif i==2
                ylabel('$Mean(K_t)$')
            end
            
            legend('Location','best');
            box('on');
            grid on;            
                    
            funs.printfig(fig);            
        end
    end       
    function [] = sim_mean_hhours(sim, par)
        
        fig = figure('Name',sprintf('mean_lhours_%s', par.prefix));
        
        labor = nan(par.Nh, par.T);
        for i = 1:par.Nh
            labor(i, :) = sum((sim.h_choice == i)) / sim.N;        
        end
        mean_hours = sum(par.h .* labor);
        plot(mean_hours, '-o', 'color', 'k', 'linewidth', 1.5, 'MarkerSize', 3);
        ylabel('$Mean(h_t)$');
        xlabel('$t$');
        grid('on');
        
        funs.printfig(fig);
        
    end
    function [] = sim_mean_hhours_vary_D(sim, sim_alternative, D, par)
        
        fig = figure('Name',sprintf('mean_lhours_%s', par.prefix));
        
        h_three = [0; 0.5; 1];
        labor = nan(3, par.T);
        for i = 1:3
            labor(i, :) = sum((sim.h_choice == i)) / sim.N;        
        end
        mean_hours = sum(h_three .* labor);
        plot(mean_hours, '-o', 'color', 'k','DisplayName','$D = 3$', 'linewidth', 1.5, 'MarkerSize', 3);
          hold on;

        plot(sim_alternative, '-o', 'color', 'r','DisplayName',sprintf('$D = %0.0f$', D), 'linewidth', 1.5, 'MarkerSize', 3);
        ylabel('$Mean(h_t)$');
        xlabel('$t$');
        legend('Location','best');
        grid('on');
        hold off;
        
        funs.printfig(fig);
        
    end
    
    % Solution figures
    function [] = cons_1d_fig(par, sol, ts, h, fix_var, fix_no, x_lim, y_lim)

        color = figs.color(); 
        
        % get right indices
        if strcmp(fix_var, 'm') == 1
            ind_m = fix_no;
            ind_k = 1:numel(par.grid_k);
            x = par.grid_k;
            xlab = '$h_t$';
        elseif strcmp(fix_var, 'h') == 1
            ind_m = 1:numel(par.grid_m);
            ind_k = fix_no;
            x = par.grid_m;
            xlab = '$m_t$';
        else
            error('choose m or h');
        end
        
        for j = 1:numel(h)
            fig = figure('Name', sprintf('cons_hours=%0.1f_t=%s_%s',...
                         par.h(h(j)), string(ts(:)), par.prefix));
                         
                % plot over t
                for i = 1:numel(ts)
                    plot(x, sol.c{ts(i), h(j)}(ind_m, ind_k), 'o',...
                         'linewidth', 1.5, 'MarkerSize', 3, 'color', color{i},...
                         'DisplayName', sprintf('t = %d', ts(i)));                               
                    hold('on');  
                end
                
                % ylabel; either h or m
                if strcmp(fix_var, 'm') == 1
                    ylabel(sprintf('$c(m_t = %0.1f, h_t, l_t = %0.1f)$', par.grid_m(fix_no), par.h(h(j))));                
                else 
                    ylabel(sprintf('$c(m_t, h_t = %0.1f, l_t = %0.1f)$', par.grid_k(fix_no), par.h(h(j))));                                    
                end
            
            % layout
            if nargin > 6
                xlim(x_lim);
                ylim(y_lim);
            end
            
            xlabel(xlab);                  
            legend('Location','best');
            box('on');
            grid on;
            
            funs.printfig(fig);                                
        end % labor choice loop                           
    end    
    function [] = value_1d_fig(par, sol, ts, h, fix_var, fix_no, x_lim, y_lim)
        
        color = figs.color(); 
        
        %Eva og Mathias:
        par.grid_k = par.grid_k(1:75);
                
        % get right indices
        if strcmp(fix_var, 'm') == 1
            ind_m = fix_no;
            ind_k = 1:numel(par.grid_k);
            x = par.grid_k;
            xlab = '$K_t$';
        elseif strcmp(fix_var, 'h') == 1
            ind_m = 1:numel(par.grid_m);
            ind_k = fix_no;
            x = par.grid_m;
            xlab = '$M_t$';
        else
            error('choose m or h');
        end
        
        for i = 1:numel(ts)
            fig = figure('Name', sprintf('value_t=%d_%s', ts(i), par.prefix));
            
                % plot over labor choice
                for j = 1:numel(h)
                    plot(x, sol.v{ts(i), h(j)}(ind_m, ind_k), 'o',...
                         'linewidth', 1.5, 'MarkerSize', 3, 'color', color{j},...
                         'DisplayName', sprintf('$h_t = %0.1f$', par.h(h(j))));                               
                    hold('on');  
                end
                
                % ylabel; either h or m
                if strcmp(fix_var, 'm') == 1
                    ylabel(sprintf('$v(M_t = %0.1f, K_t, h_t)$', par.grid_m(fix_no))); 
                else 
                    ylabel(sprintf('$v(M_t, K_t = %0.1f, h_t)$', par.grid_k(fix_no)));
                end
            
            % layout
            if nargin > 6
                xlim(x_lim);
                ylim(y_lim);
            end
            
            xlabel(xlab);
            legend('Location','best');
            box('on');
            grid on;
            
            funs.printfig(fig);                                
        end % labor time loop                                   
        
    end
    function [] = value_1d_fig_grid_compare(par, par_grid, sol, sol_grid, ts, h, fix_var, fix_no, fix_no_grid, x_lim)
        
        color = figs.color(); 
        
        % get right indices
        if strcmp(fix_var, 'm') == 1
            ind_m = fix_no;
            ind_k = 1:numel(par.grid_k);
            x = par.grid_k;
            xlab = '$K_t$';
        elseif strcmp(fix_var, 'h') == 1
            ind_m = 1:numel(par.grid_m);
            ind_k = fix_no;
            x = par.grid_m;
            xlab = '$M_t$';
        else
            error('choose m or h');
        end
        
        if strcmp(fix_var, 'm') == 1
            ind_m_grid = fix_no_grid;
            ind_k_grid = 1:numel(par_grid.grid_k);
            x_grid = par_grid.grid_k;
            xlab = '$K_t$';
        elseif strcmp(fix_var, 'h') == 1
            ind_m_grid = 1:numel(par_grid.grid_m);
            ind_k_grid = fix_no_grid;
            x_grid = par_grid.grid_m;
            xlab = '$M_t$';
        else
            error('choose m or h');
        end
        
        for i = 1:numel(ts)
            fig = figure('Name', sprintf('value_t=%d_%s', ts(i), par.prefix));
            
                % plot over labor choice
                for j = 1:numel(h)
                    plot(x, sol.v{ts(i), h(j)}(ind_m, ind_k), 'o',...
                         'linewidth', 1.5, 'MarkerSize', 3, 'color', color{j},...
                         'DisplayName', sprintf('$h_t = %0.1f, EGM $', par.h(h(j))));                               
                    hold('on');
                    plot(x_grid, sol_grid.v{ts(i), h(j)}(ind_m_grid, ind_k_grid), 'o',...
                         'linewidth', 1.5, 'MarkerSize', 3, 'color', color{j+3},...
                         'DisplayName', sprintf('$h_t = %0.1f, VFI$', par.h(h(j)))); 
                end
                
                % ylabel; either h or m
                if strcmp(fix_var, 'm') == 1
                    ylabel(sprintf('$v(M_t = %0.0f, K_t, h_t)$', par.grid_m(fix_no))); 
                else 
                    ylabel(sprintf('$v(M_t, K_t = %0.0f, h_t)$', par.grid_k(fix_no)));
                end
            
            % layout
            if nargin > 8
                xlim(x_lim);
            end
            
            xlabel(xlab);
            legend('Location','best');
            box('on');
            grid on;
            
            funs.printfig(fig);                                
        end % labor time loop                                   
        
    end
    function [] = cons_1d_fig_grid_compare(par, par_grid, sol, sol_grid, ts, h, fix_var, fix_no, fix_no_grid, x_lim)
        
        color = figs.color(); 
        
        % get right indices
        if strcmp(fix_var, 'm') == 1
            ind_m = fix_no;
            ind_k = 1:numel(par.grid_k);
            x = par.grid_h;
            xlab = '$K_t$';
        elseif strcmp(fix_var, 'h') == 1
            ind_m = 1:numel(par.grid_m);
            ind_k = fix_no;
            x = par.grid_m;
            xlab = '$M_t$';
        else
            error('choose m or h');
        end
        
         if strcmp(fix_var, 'm') == 1
            ind_m_grid = fix_no_grid;
            ind_k_grid = 1:numel(par_grid.grid_k);
            x_grid = par_grid.grid_k;
            xlab = '$K_t$';
        elseif strcmp(fix_var, 'h') == 1
            ind_m_grid = 1:numel(par_grid.grid_m);
            ind_k_grid = fix_no_grid;
            x_grid = par_grid.grid_m;
            xlab = '$M_t$';
        else
            error('choose m or h');
        end
        
        for i = 1:numel(ts)
            fig = figure('Name', sprintf('cons_t=%d_%s', ts(i), par.prefix));
            
                % plot over labor choice
                for j = 1:numel(h)
                    plot(x, sol.c{ts(i), h(j)}(ind_m, ind_k), 'o',...
                         'linewidth', 1.5, 'MarkerSize', 3, 'color', color{j},...
                         'DisplayName', sprintf('$h_t = %0.1f, EGM $', par.h(h(j))));                               
                    hold('on');
                    plot(x_grid, sol_grid.c{ts(i), h(j)}(ind_m_grid, ind_k_grid), 'o',...
                         'linewidth', 1.5, 'MarkerSize', 3, 'color', color{j+3},...
                         'DisplayName', sprintf('$h_t = %0.1f, VFI$', par.h(h(j)))); 
                end
                
                % ylabel; either h or m
                if strcmp(fix_var, 'm') == 1
                    ylabel(sprintf('$c(M_t = %0.0f, K_t, h_t)$', par.grid_m(fix_no))); 
                else 
                    ylabel(sprintf('$c(M_t, K_t = %0.0f, h_t)$', par.grid_k(fix_no)));
                end
            
            % layout
            if nargin > 6
                xlim(x_lim);
            end
            
            xlabel(xlab);
            legend('Location','best');
            box('on');
            grid on;
            
            funs.printfig(fig);                                
        end % labor time loop                                   
        
    end
    
    % Tax figures
    function [] = elasticity(elasticity, store_num_dc, x_lim, par, phi)

        fig = figure('Name',sprintf('elasticity_number_of_DC_sets_%d_%s',size(elasticity,1),par.prefix));
        
        if nargin > 4
            for i = [1:size(elasticity,1)]
                plot(elasticity(i,:), '-o', 'color', figs.color{i}, ...
                    'DisplayName', sprintf('$%s = %0.2f$', par.prefix, phi{i}), ...
                    'linewidth', 1.5, 'MarkerSize', 3);
                hold on;
            end
            
        elseif nargin < 5
            for i = [1:size(elasticity,1)]
                plot(elasticity(i,:), '-o', 'color', figs.color{i}, ...
                    'DisplayName', sprintf('$D = %0.0f$', store_num_dc(i)), ...
                    'linewidth', 1.5, 'MarkerSize', 3);
                hold on;
            end
        end
        
        xlim(x_lim);
        ylabel('Labor Supply Elasticity');
        xlabel('$t$');
        grid('on');
        legend('Location','best');
        hold off;
        
        funs.printfig(fig);
        
    end

end
end