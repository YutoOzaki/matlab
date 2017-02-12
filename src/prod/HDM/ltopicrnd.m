%% document-topic distribution
function theta = ltopicrnd(pi, alpha, n_jk, J, K)
    theta = zeros(K+1, J);

    for j=1:J
        alpha_vec = pi.*alpha;
        alpha_vec(2:(K+1),1) = alpha_vec(2:(K+1),1) + n_jk(:,j);
        
        theta(:,j) = dirichletrnd(alpha_vec);
    end
end