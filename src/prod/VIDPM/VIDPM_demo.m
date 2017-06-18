function VIDPM_demo
    clf;
    
    %% setup parameters
    K = 50;
    maxitr = 100;
    fixed = true;
    hyperprior = true;
    
    %% load data
    [x, THETA] = gentestdata();
    D = size(x, 1);
    N = size(x, 2);
    
    if fixed
        SIG = THETA.SIG;
    end
    
    %% setup parameters
    % true distributions' parameters
    alpha = 1;
    A = [1; alpha];
    B = {[0;0], 2};
    C = [5; 0.1];
    
    % parameters for variational beta distribution
    gam = repmat(A, 1, K - 1);
    
    % parameters for variational distribution of exponential family
    tau = repmat({B}, 1, K);
    
    % parameters for variational multinomial distribution
    phi = rand(K, N);
    phi = phi./repmat(sum(phi, 1), K, 1);
    
    % paramters for variational gamma distribution (hyper-prior)
    ohm = C;
    
    %% main loop
    S = zeros(K, N);
    pmt = inv(SIG);
    
    for itr=1:maxitr
        tic;
        fprintf('iteration: %d ', itr);
        
        if hyperprior
            ohm(1) = C(1) + K - 1;
            Eqlog1minV = 0;
            for i=1:K-1
                Eqlog1minV = Eqlog1minV + psi(gam(2, i)) - psi(gam(1, i) + gam(2, i));
            end
            ohm(2) = C(2) - Eqlog1minV;
            Eqalpha = ohm(1)/ohm(2);
        else
            Eqalpha = A(2);
        end

        for k=1:K
            N_k = sum(phi(k, :));
            
            gam(1, k) = A(1) + N_k;
            gam(2, k) = Eqalpha + sum(sum(phi(k+1:K, :), 1));
            tau{k}{2} = B{2} + N_k;
            tau{k}{1} = B{1} + sum(repmat(phi(k, :), D, 1).*x, 2);

            EqlogV = psi(gam(1, k)) - psi(gam(1, k) + gam(2, k));
            Eqlog1minV = 0;
            for i=1:k-1
                Eqlog1minV = Eqlog1minV + psi(gam(2, i)) - psi(gam(1, i) + gam(2, i));
            end

            mu_k = tau{k}{1}./tau{k}{2};
            Eqeta = pmt*mu_k;
            Eqa = 0.5*(D + mu_k'*pmt*mu_k);
                    
            for n=1:N
                S(k, n) = EqlogV + Eqlog1minV + Eqeta'*x(:, n) - Eqa;
            end
        end
        
        S_max = max(S, [], 1);
        S = S - repmat(S_max, K, 1);
        phi = exp(S);
        phi = phi./repmat(sum(phi, 1), K, 1);

        assert(isempty(find(isinf(phi), 1)) && isempty(find(isnan(phi), 1)), 'Inf or NaN error');

        phi = phi';
        z = mnrnd(1, phi);
        phi = phi';
        
        [row, ~] = find(z');
        idx = unique(row);
        numclass = length(idx);
        figure(2);
        for i=1:numclass
            idx_c = find(row == idx(i));
            
            color = repmat(rand(1, 3), length(idx_c), 1);
            
            scatter(x(1, idx_c), x(2, idx_c), [], color); hold on;
            mu_c = tau{idx(i)}{1}./tau{idx(i)}{2};
            scatter(mu_c(1), mu_c(2), [], ...
                'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color(1, :), 'LineWidth', 1.0); hold on;
            
            drawellip(mu_c, SIG); hold on
        end
        title(sprintf('C = %d', numclass));
        hold off;
        drawnow;
    
        t = toc;
        fprintf('(elapsed time %3.3f sec)\n', t);
    end
end

function [x, THETA] = gentestdata()
    D = 2;
    K = randi(10);
    
    df = 6;
    PSI = diag(rand(D, 1) .* 10);
    SIG = iwishrnd(PSI, df);
    
    m = 1/0.02;
    mu_0 = rand(D, 1);
    mu = mvnrnd(mu_0, m.*SIG, K)';
    
    n = zeros(K, 1);
    for k=1:K
        n(k) = 10 + randi(50);
    end
    N = sum(n);
    
    x = zeros(D, N);
    idx_e = 0;
    figure(1);
    for i=1:K
        tmp = mvnrnd(mu(:, i), SIG, n(i));
        idx_s = idx_e + 1;
        idx_e = idx_e + n(i);
        x(:, idx_s:idx_e) = tmp';
        
        scatter(tmp(:, 1), tmp(:, 2)); hold on
        scatter(mu(1, i), mu(2, i), [], 'MarkerEdgeColor', [0 0 0], 'LineWidth', 1.0); hold on;
        drawellip(mu(:, i), SIG); hold on
    end
    hold off;
    title(sprintf('K = %d', K));
    drawnow;
    
    THETA = struct('K', K, 'df', df, 'PSI', PSI,...
        'SIG', SIG, 'm', m, 'mu_0', mu_0, 'mu', mu);
end