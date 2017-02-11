%% topic-word distribution
function phi = twordrnd(V, K, n_kv, beta)
    phi = zeros(V, K);
    
    for k=1:K
        phi(:,k) = dirichletrnd(n_kv(k, :) + beta);
    end
end