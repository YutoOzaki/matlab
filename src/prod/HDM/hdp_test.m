function hdp_test(testdata, gamma, a_gam, b_gam, alpha, a_alpha, b_alpha, beta, steps, steps_hp, steps_fpi, maxitr, scheme)
    %% Load dataset
    load(testdata);
    assert(exist('N', 'var') && exist('V', 'var') && exist('vocabulary', 'var') && exist('w', 'var'),...
        'data mat file should contain N, V, vocabulary and w');
    
    for i=1:N
        if iscell(w{i})
            w{i} = cell2mat(w{i});
        end
    end
    
    repository = datarepo(w, N, V, vocabulary);
    
   %% Setup test condition
    L = zeros(N, maxitr);
    perplexity = zeros(maxitr, 1);
    bufperp = Inf;
    
    diary off
    
    %% CRF initialization
    repository = crf(repository, alpha, gamma, beta, true, 1);
    pi = dirichletrnd([gamma; repository.m_k]);
    theta = ltopicrnd(pi, alpha, repository.n_jk, repository.J, repository.K);
    
    %% CRF inference
    for i=1:maxitr
        if strcmp('direct', scheme)
            repository = directassignment(repository, alpha, gamma, beta, pi, false, steps);
        elseif strcmp('crf', scheme)
            repository = crf(repository, alpha, gamma, beta, false, steps);
        elseif strcmp('prs', scheme)
            repository = prs(repository, alpha, gamma, beta, pi, theta, false, steps);
        end
        
        %{
        [n_jk, n_k_check] = count_jk(topicMat, N, K);
        assert(isequal(n_k, n_k_check), 'n_k count is inconsistent');
        
        K_check = count_k(topicMat);
        assert(K == K_check, 'K count is inconsistent');

        [n_kv_check] = count_kv(topicMat, w, K, V);
        assert(isequal(n_kv, n_kv_check), 'n_kv count is inconsistent');
        %}
        
        K = repository.K;
        M = repository.M;
        n_j = repository.n_j;
        n_jk = repository.n_jk;
        n_kv = repository.n_kv;
        m_k = repository.m_k;
        
        pi = dirichletrnd([gamma; m_k]);
        theta = ltopicrnd(pi, alpha, n_jk, N, K);
        phi = twordrnd(V, K, n_kv, beta);
        
        drawdist(pi, theta, phi, 3);
        [perp, loglik] = wordperp(theta(2:end,:), phi, w, N, K, n_j);
        perplexity(i) = perp;
        L(:,i) = loglik;
        
        if perp < bufperp
            bestprm.phi = phi;
            bestprm.theta = theta;
            bestprm.pi = pi;
            bestprm.alpha = alpha;
            bestprm.gamma = gamma;
            bestprm.K = K;
            
            bufperp = perp;
        end
        
        figure(5);
        subplot(2,1,1);imagesc(L(:,1:i));colorbar;
        subplot(2,1,2);plot(perplexity(1:i));
        
        fprintf(' alpha = %3.3f, gamma = %3.3f, beta = %3.3f, K = %d, M = %d, perplexity = %3.3f\n\n'...
            , alpha, gamma, beta, K, M, perplexity(i));
        
        fprintf('-before-\n');
        K_mean =expcrp(gamma, M);
        M_mean = mean_m(alpha, n_j);
        fprintf(' E(K) = %3.3f (gam = %3.3f, K = %d)\n', K_mean, gamma, K);
        fprintf(' E(M) = %3.3f (alpha = %3.3f, M = %d)\n', M_mean, alpha, M);
        fprintf(' beta = %3.3f\n', beta);

        [gamma, alpha] = hyprprmrnd(K, M, gamma, alpha, n_j, a_gam, b_gam, a_alpha, b_alpha, steps_hp);
        n_k  = repository.n_k;
        beta = fixeditr(n_kv, n_k, beta, V, steps_fpi);

        fprintf('-after-\n');
        K_mean = expcrp(gamma, M);
        M_mean = mean_m(alpha, n_j);
        fprintf(' E(K) = %3.3f (gam = %3.3f, K = %d)\n', K_mean, gamma, K);
        fprintf(' E(M) = %3.3f (alpha = %3.3f, M = %d)\n', M_mean, alpha, M);
        fprintf(' beta = %3.3f\n', beta);
    end
    diary on

    %% Best perplexity
    [minper, idx] = min(perplexity);
    fprintf('best perplexity %3.3f at iteration No.%d\n', minper, idx);
    [perp, ~] = wordperp(groundtruth.theta, groundtruth.H, w, N, groundtruth.K, n_j);
    fprintf('ground truth perplexity %3.3f\n', perp);
    
    %% Compare with ground truth
    hdp_testdata_result(bestprm.K, bestprm.gamma, bestprm.alpha, beta, bestprm.pi, bestprm.theta, bestprm.phi, groundtruth);
end