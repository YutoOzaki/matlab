function pi = hdpposteriorWrapper(pi, repo, alpha, z)
    L = length(alpha);
    
    N = size(z, 1);
    z = [ones(N, 1) z];
    
    for l=L:-1:2
        pi{l} = hdpposteriorHelper(pi{l}, pi{l - 1}, repo, alpha(l));
        
        if l > 2
            repo = countHelper(z(:, (l - 2):(l - 1)), pi{l - 1}, alpha(l), repo);
        end
    end
     
    m_k = count_mk(alpha(1), pi{1}, repo.n_jkSet{1});
    pi{1} = dirichletrnd([m_k; alpha(1)]);
end

function pi = hdpposteriorHelper(pi, beta, repo, alpha)
    pi = pi .* 0;
    G = size(beta, 2);
    
    for g=1:G
        beta_g = beta(:, g);
        n_jk = repo.n_jkSet{g};
        idx = repo.idxSet{g};

        pi_update = hdpposterior(alpha, beta_g, n_jk);
        pi(:, idx) = pi_update;
    end
end

function repo_u = countHelper(z, beta, alpha, repo)
    K = size(beta, 1) - 1;
    J = length(unique(z));
    n_jk = zeros(J, K);
    
    for j=1:J
        n_jk(j, :) = count_mk(alpha, beta(:, j), repo.n_jkSet{j});
    end
    
    G = length(unique(z(:, 1)));
    n_jkSet = cell(G, 1);
    idxSet = cell(G, 1);
   
    for g=1:G
        idx = z(:, 1) == g;
        j = unique(z(idx, 2));
        
        n_jkSet{g} = n_jk(j, :);
        idxSet{g} = j;
    end
    
    repo_u = struct('idxSet', {idxSet}, 'n_jkSet', {n_jkSet});
end

function m_k = count_mk(alpha, beta, n_jk)
    K = length(beta) - 1;
    J = size(n_jk, 1);
    
    m_k = zeros(K, 1);
    
    for k=1:K
        alphabeta = alpha .* beta(k);
        
        for j=1:J
            if n_jk(j, k) > 0
                z = zeros(n_jk(j, k), 1);
                [~, ~, m_jk] = crp(alphabeta, z);
                m_k(k) = m_k(k) + m_jk;
            end
        end
    end
end