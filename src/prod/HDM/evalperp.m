function [perplexity, loglik] = evalperp(x, pi, phi)
    N = length(x);
    K = size(pi, 1) - 1;
    loglik = zeros(N, 1);
    
    n_j = cell2mat(cellfun(@length, x, 'UniformOutput', false));

    pi_tmp = pi(1:K, :)';
    
    for j=1:N
        for i=1:n_j(j)
            v = x{j}(i);

            tmp = pi_tmp(j, :) * phi(1:K, v);
            loglik(j) = loglik(j) + log(tmp);
        end
    end
    
    perplexity = exp(-sum(loglik)/sum(n_j));
end