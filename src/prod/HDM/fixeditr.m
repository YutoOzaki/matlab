function beta = fixeditr(n_kv, n_k, beta, V, steps)
    K = length(n_k);
    KV = K*V;
    
    for i=1:steps
        num = sum(sum(psi(n_kv + beta))) - KV*psi(beta);
        den = V*sum(psi(n_k + beta*V)) - KV*psi(beta*V);
        
        beta_new = beta * num / den;
        
        beta = beta_new;
    end
end