%% topic-word distribution
function phi = worddist(V, K, n_kv, beta)
    phi = zeros(V, K + 1);
    
    for k=1:K
        phi(:, k) = dirichletrnd(n_kv(k, :) + beta);
    end
    
    phi(:, K + 1) = dirichletrnd(ones(1, V) .* beta);
    
    phi = phi';
end