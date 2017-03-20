function [z, n_k, K] = crp(alpha, z)
    z = z(:);
    
    utils.checkInteger(z, 'z');
    
    if isempty(find(z < 1, 1))
        k = unique(z);
        utils.checkSequence(k, 'k');
        assert(min(z) == 1, 'Indexing should begin from 1');
        
        K = length(k);
        init = false;
    else
        N = length(z);
        zero = zeros(N, 1);
        assert(isequal(z, zero), 'The index vector should only be zero if not initialized yet');
        
        K = 0;
        init = true;
    end
    
    n_k = zeros(K + 1, 1);
    for k=1:K
        n_k(k) = length(find(z == k));
    end
    n_k(K + 1) = alpha;
    
    if init
        [z, n_k, K] = crp_init(alpha, z, n_k, K);
    else
        [z, n_k, K] = crp_resample(alpha, z, n_k, K);
    end
    
    n_k = n_k(1:K);
end

function [z, n_k, K] = crp_init(alpha, z, n_k, K)
    N = length(z);
    
    for n=1:N
        p = n_k./(n - 1 + alpha);
        z_n = find(mnrnd(1,p));
        
        if z_n == (K + 1)
            n_k = utils.insertElement(n_k, 1, K + 1);
            K = K + 1;
        else
            n_k(z_n) = n_k(z_n) + 1;
        end
        
        z(n) = z_n;
    end
end

function [z, n_k, K] = crp_resample(alpha, z, n_k, K)
    N = length(z);
    
    for n=1:N
        z_n = z(n);
        n_k(z_n) = n_k(z_n) - 1;
        
        if n_k(z_n) == 0
            n_k = utils.deleteElement(n_k, z_n);
            K = K - 1;
            
            idx = z > z_n;
            z(idx) = z(idx) - 1;
        end
        
        p = n_k./(N - 1 + alpha);
        z_n = find(mnrnd(1,p));
        
        if z_n == (K + 1)
            n_k = utils.insertElement(n_k, 1, K + 1);
            K = K + 1;
        else
            n_k(z_n) = n_k(z_n) + 1;
        end
        
        z(n) = z_n;
    end
end
    

