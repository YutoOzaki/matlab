function pi = hdpposterior(alpha, beta, n_jk)
    K = length(beta) - 1;
    J = size(n_jk, 1);
    
    pi = zeros(K + 1, J);
    
    n_jk(:, K + 1) = 0;
    alphabeta = alpha .* beta;
    alphabeta = alphabeta(:)';
    
    for j=1:J
        pi(:, j) = dirichletrnd(alphabeta + n_jk(j, :));
    end
end