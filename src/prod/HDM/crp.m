function [K, count_n] = crp(alpha, N)
    K = 0;
    count_n = zeros(K+1, 1);
    count_n(1) = alpha;
    
    for n=1:N
        p = count_n./(n-1+alpha);
        z = find(mnrnd(1,p));
        
        if z == (K+1)
            count_n(z) = 1;
            count_n = [count_n; alpha];
            
            K = K + 1;
        else
            count_n(z) = count_n(z) + 1;
        end
    end
end