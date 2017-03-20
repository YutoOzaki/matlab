function pi = sbptree(alpha, K, z)
    L = length(alpha);
    pi = cell(L, 1);
    N = size(z, 1);
    
    pi{1} = sbp(alpha(1), K);
    
    z = [ones(N, 1) z (1:N)'];
    
    for l=2:L
        J = length(unique(z(:, l)));
        pi_tmp = zeros(K, J);
        
        for j=1:J
            idx = find(z(:, l) == j);
            beta = pi{l - 1}(:, z(idx(1), l - 1));
            
            pi_tmp(:, j) = hsbp(alpha(l), beta);
        end
        
        pi{l} = pi_tmp;
    end
end