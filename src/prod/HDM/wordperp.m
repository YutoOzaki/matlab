function [perp, L] = wordperp(theta, phi, w, J, K, n_j)
    L = zeros(J, 1);

    for j=1:J
        for n=1:n_j(j)
            L_buf = 0;
            for k=1:K
                L_buf = L_buf + theta(k,j) * phi(w{j}(n),k);
            end
            L(j) = L(j) + log(L_buf);
        end
    end

    perp = exp(-sum(L)/sum(n_j));
end