function z = ncrp(eta, z)
    if isempty(eta)
        z = zeros(size(z, 1), 0);
    else
        c = size(z, 2);
        
        if c == 1
            z = ncrp_init(eta, z);
        else
            L = length(eta);
            assert(L == c, 'array size of z and eta is unexpected');

            z = ncrp_resample(eta, z);
        end
        
        z = numbering(z);
    end
end

function z = ncrp_init(eta, z)
    [z, n_k] = crp(eta(1), z);

    N = size(z, 1);
    z = [z zeros(N, length(eta) - 1)];

    eta = eta(2:end);

    if ~isempty(eta)
        K = length(n_k);
        
        for k=1:K
            idx = find(z(:, 1) == k);
            z_tmp = zeros(length(idx), 1);
            
            z_l = ncrp_init(eta, z_tmp);
            z(idx, 2:end) = z_l;
        end
    end
end

function z = ncrp_resample(eta, z)
    
    [z_tmp, n_k] = crp(eta(1),  z(:, 1));
    
    z(:, 1) = z_tmp;

    eta = eta(2:end);
    
    if ~isempty(eta)  
        K = length(n_k);
        
        for k=1:K
            idx = find(z(:, 1) == k);
            z_tmp = z(idx, 2);
            
            k_tmp = unique(z_tmp);
            K_tmp = length(k_tmp);
            for kk=1:K_tmp
                idx_tmp = z_tmp == k_tmp(kk);
                z_tmp(idx_tmp) = kk;
            end
            z(idx, 2) = z_tmp;
            
            z_l = ncrp_resample(eta, z(idx, 2:end));
            z(idx, 2:end) = z_l;
        end
    end
end

function z = numbering(z)
    c = size(z, 2);
    
    if c > 1
        S = length(unique(z(:, 1)));

        C = 0;
        for s=1:S
            idx = z(:, 1) == s;

            z(idx, 2) = z(idx, 2) + C;

            C = C + length(unique(z(idx, 2)));
        end
        
        z_tmp = z(:, 2:end);
        z(:, 2:end) = numbering(z_tmp);
    end
end