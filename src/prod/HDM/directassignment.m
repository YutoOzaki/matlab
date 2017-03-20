function [pi, n_jk, n_kv, n_k, z_ji] = directassignment(x, alpha, pi, f_k, stickbreaking, idxset, repo, steps)
    if nargin == 6
        repo = [];
        steps = 0;
    elseif nargin == 7
        steps = 0;
    end
    
    n_j = cell2mat(cellfun(@length, x, 'UniformOutput', false));
    
    if isempty(repo)
        [pi, n_jk, n_kv, n_k, z_ji] = directassignment_init(x, alpha, pi, f_k, stickbreaking, idxset);
    else
        n_k = repo.n_k;
        n_kv = repo.n_kv;
        n_jk = repo.n_jk;
        z_ji = repo.z_ji;
        
        [pi, n_jk, n_kv, n_k, z_ji] = directassignment_resample(x, alpha, pi, f_k, stickbreaking, idxset, n_k, n_kv, n_jk, z_ji);
    end

    K = length(pi{1}) - 1;
    assert(length(n_k) == K, 'Length of n_k and K is inconsistent');
    assert(size(n_kv, 1) == K && size(n_kv, 2) == 1/f_k(0, 0), 'Size of n_kv, K and V is inconsistent');
    assert(sum(n_k) == sum(sum(n_kv)) && sum(n_k) == sum(sum(n_jk)) && sum(n_j) == sum(n_k), 'Count of n_k, n_kv and n_jk is inconsistent');
    
    for itr=1:steps
        [pi, n_jk, n_kv, n_k, z_ji] = directassignment_resample(x, alpha, pi, f_k, stickbreaking, idxset, n_k, n_kv, n_jk, z_ji);
        
        K = length(pi{1}) - 1;
        assert(length(n_k) == K, 'Length of n_k and K is inconsistent');
        assert(size(n_kv, 1) == K && size(n_kv, 2) == 1/f_k(0, 0), 'Size of n_kv, K and V is inconsistent');
        assert(sum(n_k) == sum(sum(n_kv)) && sum(n_k) == sum(sum(n_jk)) && sum(n_j) == sum(n_k), 'Count of n_k, n_kv and n_jk is inconsistent');
    end
end

function [pi, n_jk, n_kv, n_k, z_ji] = directassignment_init(x, alpha, pi, f_k, stickbreaking, idxset)
    betaSet = pi{end - 1};
    K = size(betaSet, 1) - 1;
    
    J = length(x);
    n_j = cell2mat(cellfun(@length, x, 'UniformOutput', false));
    
    p = zeros(K + 1, 1);
    
    n_k = zeros(K, 1);
    n_kv = zeros(K, 1/f_k(0, 0));
    n_jk = zeros(J, K);
    z_ji = x;
    
    for j=1:J
        beta = betaSet(:, idxset(j));
        
        for i=1:n_j(j)
            v = x{j}(i);
            
            for k=1:K
                p(k) = (n_jk(j, k) + alpha*beta(k)) * f_k(n_kv(k, v), n_k(k));
            end

            p(K + 1) = alpha*beta(K + 1) * f_k(0, 0);

            p = p./sum(p);
            z = find(mnrnd(1, p));

            if z == (K + 1)
                pi = stickbreaking(pi);
                betaSet = pi{end - 1};
                beta = betaSet(:, idxset(j));

                K = K + 1;

                p(K + 1, 1) = 0; 
                n_k(K, 1) = 1;
                n_kv(K, v) = 1;
                n_jk(j, K) = 1;
            else
                try
                    n_k(z, 1) = n_k(z, 1) + 1;
                catch e
                    fprintf('hey');
                end
                n_kv(z, v) = n_kv(z, v) + 1;
                n_jk(j, z) = n_jk(j, z) + 1;
            end
            
            z_ji{j}(i) = z;
        end
    end
end

function [pi, n_jk, n_kv, n_k, z_ji] = directassignment_resample(x, alpha, pi, f_k, stickbreaking, idxset, n_k, n_kv, n_jk, z_ji)
    betaSet = pi{end - 1};
    K = size(betaSet, 1) - 1;
    
    J = length(x);
    n_j = cell2mat(cellfun(@length, x, 'UniformOutput', false));
    
    p = zeros(K + 1, 1);
    
    for j=1:J
        beta = betaSet(:, idxset(j));
        
        for i=1:n_j(j)
            v = x{j}(i);
            k = z_ji{j}(i);
            
            n_k(k) = n_k(k) - 1;
            n_kv(k, v) = n_kv(k, v) - 1;
            n_jk(j, k) = n_jk(j, k) - 1;
            
            if n_k(k) == 0
                assert(n_kv(k, v) == 0 && n_jk(j, k) == 0, 'count of n_k, n_kv and n_jk is inconsistent');
                
                n_k = utils.deleteElement(n_k, k);
                n_kv = utils.deleteElement(n_kv, k);
                n_jk = utils.deleteElement(n_jk', k)';
                p = utils.deleteElement(p, k);
                
                pi = utils.deleteElement(pi, k);
                betaSet = pi{end - 1};
                beta = betaSet(:, idxset(j));
                
                idx = cellfun(@(x) find(x >k), z_ji, 'UniformOutput', false);
                for jj=1:J
                    z_ji{jj}(idx{jj}) = z_ji{jj}(idx{jj}) - 1;
                end

                K = K - 1;
            end
            
            for k=1:K
                p(k) = (n_jk(j, k) + alpha*beta(k)) * f_k(n_kv(k, v), n_k(k));
            end

            p(K + 1) = alpha*beta(K + 1) * f_k(0, 0);

            p = p./sum(p);
            z = find(mnrnd(1, p));

            if z == (K + 1)
                pi = stickbreaking(pi);
                betaSet = pi{end - 1};
                beta = betaSet(:, idxset(j));

                K = K + 1;

                p(K + 1, 1) = 0; 
                n_k(K, 1) = 1;
                n_kv(K, v) = 1;
                n_jk(j, K) = 1;
            else
                n_k(z, 1) = n_k(z, 1) + 1;
                n_kv(z, v) = n_kv(z, v) + 1;
                n_jk(j, z) = n_jk(j, z) + 1;
            end
            
            z_ji{j}(i) = z;
        end
    end
end