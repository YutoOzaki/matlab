function pi = hsbp(alpha, beta)
    K = length(beta);
    logpi = zeros(K, 1);
    
    residual = log(1);
    partition = 1;
    
    for k=1:K-1
        partition = partition - beta(k);
        
        pi_tmp = betarnd(alpha*beta(k), alpha*partition);
        
        if isnan(pi_tmp)
            pi_tmp = betarnd(eps, eps);
        end
        
        logpi(k) = log(pi_tmp) + residual;
        residual = residual + log(1 - pi_tmp);
    end
    
    pi = exp(logpi);
    
    pi(K) = 1 - sum(pi(1:k));
    
    while pi(K) < 0
        warning('off', 'backtrace');
        warning('hsbp(alpha, beta) >> An overflow has occured: beta(K) = %e', beta(K));
        
        val = pi(K);
        
        [~, I] = max(pi);
        pi(I) = pi(I) + val;
        
        pi(K) = 1 - sum(pi(1:k));
    end
end