function hdp_testdata(K, J, V, gamma, alpha, beta, n_j_min, n_j_max, converged, INCREMENTAL)
    margin = n_j_max - n_j_min;
    
    %% generate base measure
    pi = sbp(gamma, K);
    
    %% generate children Dirichlet process
    theta = zeros(K, J);
    for j=1:J
        theta(:,j) = hsbp(alpha, K, pi);
    end
    
    %% generate distribution of each factors
    H = zeros(V, K);
    for k=1:K
        H(:,k) = dirichletrnd(beta .* ones(V,1));
    end
    
    %% generate each document's words length
    n_j = zeros(J, 1);
    for j=1:J
        n_j(j) = n_j_min + randi(margin);
    end
    
    %% see expectation of M and K under current hyperparameters
    M_mean = mean_m(alpha, n_j);
    K_mean = expcrp(gamma, M_mean);
    
    %% generate words
    w = cell(J, 1);
    loop = 1;
    while loop <= converged
        for j=1:J
            w_buf = zeros(n_j(j), 1);
            n = 1;

            while n <= n_j(j)
                k = mnrnd(1, theta(:,j));
                idx = find(k == 1);
                
                if INCREMENTAL == true && idx == K
                    mu_0 = betarnd(1, gamma);
                    pi_new_0 = pi(K) * (1 - mu_0);
                    pi_new_K = pi(K) * mu_0;

                    pi_new = zeros(K+1,1);
                    pi_new(K+1) = pi_new_0;
                    pi_new(1:K-1) = pi(1:K-1);
                    pi_new(K) = pi_new_K;

                    mu_d = betarnd(alpha*pi(K)*mu_0, alpha*pi(K)*(1 - mu_0), 1, J);
                    theta_new_0 = theta(K,:).*(1 - mu_d);
                    theta_new_K = theta(K,:).*mu_d;

                    theta_new = zeros(K+1,J);
                    theta_new(K+1,:) = theta_new_0;
                    theta_new(1:K-1,:) = theta(1:K-1,:);
                    theta_new(K,:) = theta_new_K;

                    pi = pi_new;
                    theta = theta_new;
                    H = [H dirichletrnd(beta .* ones(V,1))];

                    K = K + 1;
                    
                    loop = 0;
                else
                    v = mnrnd(1, H(:, idx));
                    w_buf(n) = find(v == 1);

                    n = n + 1;
                end
            end

            w{j} = w_buf;
        end
        
        loop = loop + 1;
    end
    
    %% See word histogram of each document
    whist = zeros(V, J);
    for j=1:J
        whist(:,j) = wordhist(w{j}, V);
    end
    
    assert(abs(sum(pi) - 1) < 1e-10, 'probability vector pi is corrupted');
    assert(isempty(find(abs(sum(theta,1) - 1) > 1e-10, 1)), 'probability vector theta is corrupted');
    
    figure(1);
        subplot(1,12,1); imagesc(pi); caxis([0 1]); set(gca, 'XTick', []); title('global-level');
        subplot(1,12,3:12); imagesc(theta); caxis([0 1]); title('local-level');
    
    figure(2);
        imagesc(H); caxis([0 1]); title('topic-word distribution');
        
    figure(3);
        imagesc(whist); caxis([0 1]); title('document-word histogram');
    
    figure(4);
        stem(pi, 'Marker', 'None');set(gca, 'XTick', []);
        
    vocabulary = cell(V, 1);
    for v=1:V
        vocabulary{v} = v;
    end
    
    N = J;
    groundtruth.K = K;
    groundtruth.pi = pi;
    groundtruth.theta = theta;
    groundtruth.H = H;
    groundtruth.gamma = gamma;
    groundtruth.alpha = alpha;
    groundtruth.beta = beta;
    groundtruth.M_mean = M_mean;
    groundtruth.K_mean = K;
    groundtruth.INCREMENTAL = INCREMENTAL;
    save('testdata_hdp.mat', 'N', 'V', 'vocabulary', 'w', 'groundtruth');
end

function hist = wordhist(w, V)
    hist = zeros(V, 1);
    
    for v=1:V
        hist(v) = length(find(w == v));
    end
    
    hist = hist./length(w);
end

function pi = hsbp(alpha, K, beta)
    pi = zeros(K, 1);
    
    residual = 1;
    
    for k=1:K-1
        pi_buf = betarnd(alpha*beta(k), alpha*(1 - sum(beta(1:k))));
        
        pi(k) = pi_buf * residual;
        residual = residual * (1 - pi_buf);
    end
    
    pi(K) = 1 - sum(pi);
    
    assert(isempty(find(pi < 0, 1)), 'probability vector contains minus');
end

function beta = sbp(gamma, K)
    beta = zeros(K, 1);

    beta_buf = betarnd(1, gamma, [K,1]);
    residual = 1;
    
    for k=1:K-1
       beta(k) = beta_buf(k) * residual;
       residual = residual * (1 - beta_buf(k));
    end
    
    beta(K) = 1 - sum(beta);
    
    assert(isempty(find(beta < 0, 1)), 'probability vector contains minus');
end