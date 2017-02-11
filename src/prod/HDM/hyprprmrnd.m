function [gam, alpha] = hyprprmrnd(K, M, gam, alpha, n_j, a_gam, b_gam, a_alpha, b_alpha, maxitr)
    if ~exist('maxitr','var')
        maxitr = 50;
    end

    J = length(n_j);
    I = ones(J,1);
    
    for i=1:maxitr
        w_j = betarnd((gam+1), M);
        B = b_gam - log(w_j);
        s_j = binornd(1, M/(gam + M));
        A = a_gam + K - s_j;

        gam = gamrnd(A, 1/B);

        w_j = betarnd(repmat(alpha+1,J,1), n_j);
        B = b_alpha - sum(log(w_j));
        s_j = binornd(I, n_j./(alpha + n_j));
        A = a_alpha + M - sum(s_j);

        alpha = gamrnd(A, 1/B);
    end
end