function pi = stickbreaking(alpha, pi, z)
    K = length(pi{1});
    
    %% parent node
    mu_0 = betarnd(1, alpha(1));
        
    pi_K_new = pi{1}(K, 1) * mu_0;
    pi_0_new = pi{1}(K, 1) * (1 - mu_0);
   
    beta_0 = pi{1}(K, 1);
    pi{1}(K, 1) = pi_K_new;
    pi{1}(K + 1, 1) = pi_0_new;
    
    %% children nodes
    N = size(z, 1);
    z = [ones(N, 1) z (1:N)'];
    L = length(pi);
    
    for l=2:L
        J = length(unique(z(:, l)));
        mu_d = zeros(J, 1);
        beta_d = zeros(J, 1);
        
        for j=1:J
            idx = find(z(:, l) == j, 1);
            I = z(idx, l - 1);
            
            a = alpha(l)*beta_0(I) * mu_0(I);
            b = alpha(l)*beta_0(I) * (1 - mu_0(I));
            
            if a == 0 && b == 0
                mu_d(j) = betarnd(eps, eps);
            else
                mu_d(j) = betarnd(a, b);
            end
            
            pi_K_new = pi{l}(K, j) * mu_d(j);
            pi_0_new = pi{l}(K, j) * (1 - mu_d(j));
            
            beta_d(j) = pi{l}(K, j);
            pi{l}(K, j) = pi_K_new;
            pi{l}(K + 1, j) = pi_0_new;
        end
        
        mu_0 = mu_d;
        beta_0 = beta_d;
    end
end