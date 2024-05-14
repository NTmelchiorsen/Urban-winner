classdef funs
methods(Static)
    
    function [] = layout()
        
        % set layout parameters
        set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
        set(groot, 'defaultLegendInterpreter','latex');
        set(groot, 'defaultTextInterpreter','latex');
        set(groot, 'defaultAxesFontSize', 12); 
        
    end    
    function [x, w] = GaussHermite(n)

        i   = 1:n-1;
        a   = sqrt(i/2);
        CM  = diag(a,1) + diag(a,-1);
        [V, L]   = eig(CM);
        [x, ind] = sort(diag(L));
        V       = V(:,ind)';
        w       = sqrt(pi) * V(:,1).^2;

    end
    function [x, w] = GaussHermite_lognorm(sigma,n)
        
        [x,w] = funs.GaussHermite(n);
                
        x = exp(x*sqrt(2)*sigma-0.5*sigma^2); % mu = -0.5*sigma^2
        w = w./sqrt(pi);
        
        % assert a mean of one
        assert(1-sum(w.*x) < 1e-8) % we have parametrized the distribution to have mean 1
        
    end
    function x = nonlinspace(lo,hi,n,phi)
        % recursively constructs an unequally spaced grid.
        % phi > 1 -> more mass at the lower end of the grid.
        % lo can be a vector (x then becomes a matrix).

        x      = NaN(n,length(lo));
        x(1,:) = lo;
        for i = 2:n
            x(i,:) = x(i-1,:) + (hi-x(i-1,:))./((n-i+1)^phi);
        end

    end
    function [] = printfig(figin)

        fig = figure(figin);
        fig.PaperUnits = 'centimeters';   
        fig.PaperPositionMode = 'manual';
        fig.PaperPosition = [0 0 16 12];
        fig.PaperSize = [16 12];

        filename = ['figs\' get(fig,'name') ''];            
        print('-dpdf',['' filename '.pdf']);
        

    end
    function [log_sum, prob] = logsum(v1,v2,sigma) % v1 and v2 are allowed to be column-vectors with same size
        % calculates the log-sum and choice-probabilities.

        % 1. setup
        V           = [v1,v2]; % stack them next to each other
        DIM         = size(v1,1); % note that v1 and v2 has to be column vectors

        % 2. maximum over the discrete choices
        [mxm,id]    = max(V,[],2); % mxm is a column vector with the max of each row. id the column index of the max
        
        % 3. logsum and probabilities
        if abs(sigma) > 1.0e-10

            % a. numerically robust log-sum
            log_sum  = mxm + sigma*log(sum(exp((V-mxm*ones(1,2))./sigma),2));

            % b. numerically robust probability
            prob = exp((V-log_sum*ones(1,2))./sigma);
            % note that the denominator is simply exp(log_sum./sigma). Multiply with ones(1,2) to get matrix with 2 columns
            % note that the numerator is simply exp(V./sigma)
            % finally note that the minus is due to V = log(v./sigma) and log(denominator) = log_sum./sigma
            % since log(a / b) = log(a) - log(b) and exp(log(a / b)) = exp(log(a) - log(b))
        
        else % no smoothing -> max-operator

            log_sum  = mxm; % pick the max

            prob    = zeros(DIM,2); % prob is [1,0] or [0,1]
            I       = cumsum(ones(DIM,1)) + (id-1)*DIM; % calculate linear index
            % note cumsum(ones(DIM,1)) = (1:DIM)'
            prob(I) = 1;
        end % if no shock
    end % logsum
    
    function [LogSum, Prob] = logsum_2(v1,v2,v3,sigma)% three labour choices - for gridsearch
        % calculates the log-sum and choice-probabilities.

        % 1. setup
        V           = [v1,v2,v3];
        DIM         = size(v1,1);

        % 2. maximum over the discrete choices
        [mxm,id]    = max(V,[],2);

        % 3. logsum and probabilities
        if abs(sigma) > 1.0e-10

            % a. numerically robust log-sum
            LogSum  = mxm + sigma*log(sum(exp((V-mxm*ones(1,3))./sigma),2));

            % b. numerically robust probability
            Prob = exp((V-LogSum*ones(1,3))./sigma);

        else % no smoothing -> max-operator

            LogSum  = mxm;

            Prob    = zeros(DIM,2);
            I       = cumsum(ones(DIM,1)) + (id-1)*DIM; % calculate linear index
            Prob(I) = 1;

        end

    end 
    
    function [log_sum, prob] = logsum_vec(V, sigma) % V is a matrix with column vectors stacked next to each other
        % calculates the log-sum and choice-probabilities.

        % 1. setup
        rows         = size(V, 1);
        cols         = size(V, 2);

        % 2. maximum over the discrete choices
        [mxm, id]    = max(V, [], 2); % mxm is a column vector with the max of each row. id the column index of the max
        
        % 3. logsum and probabilities
        if abs(sigma) > 1.0e-10

            % a. numerically robust log-sum
            log_sum  = mxm + sigma*log(sum(exp((V-mxm*ones(1, cols))./sigma),2));

            % b. numerically robust probability
            prob = exp((V - log_sum*ones(1, cols))./sigma);
            
        else % no smoothing -> max-operator

            log_sum  = mxm; % pick the max
            prob    = zeros(rows, cols); % prob is 1 in one of the cols
            I       = cumsum(ones(rows, 1)) + (id-1)*rows; % calculate linear index
            prob(I) = 1;
        end % if no shock
        
        % if we have -inf in the matrix V
        log_sum(isnan(log_sum)) = -Inf;
        prob(isnan(prob)) = 1/cols;
    end % logsum    
end
end

