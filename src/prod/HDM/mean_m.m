function M_mean = mean_m(alpha, n_j)
    M_mean = 0;
    J = length(n_j);
    
    for j=1:J
        M_mean = M_mean + expcrp(alpha, n_j(j));
    end
end