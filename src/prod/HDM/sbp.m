function beta = sbp(gamma, K)
    logbeta = zeros(K, 1);

    beta_tmp = betarnd(1, gamma, [K 1]);
    residual = log(1);
    
    for k=1:K-1
       logbeta(k) = log(beta_tmp(k)) + residual;
       residual = residual + log(1 - beta_tmp(k));
    end
    
    beta = exp(logbeta);
    
    beta(K) = 1 - sum(beta(1:k));
    
    while beta(K) < 0
        warning('off', 'backtrace');
        warning('sbp(gamma, K) >> An overflow has occured: beta(K) = %e', beta(K));
        
        val = beta(K);
        
        [~, I] = max(beta);
        beta(I) = beta(I) + val;
        
        beta(K) = 1 - sum(beta(1:k));
    end
end