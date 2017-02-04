function crf_tm_poc(testdata)
    %% Setup logging
    if strcmp(computer, 'PCWIN64')
        slash = '\';
    elseif strcmp(computer, 'MACI64')
        slash = '/';
    end
    
    logfilename = strcat('result_rev5_', sprintf('%5.3f',now),'.txt'); 
    homedir = userpath;
    homedir = homedir(1:(length(homedir) - 1));
    logging = strcat(homedir,slash,'logs',slash,'HDM',slash,logfilename);
    diary(logging);
    fprintf('*** Experiment %5.5f ***\n', now);
    
    %% Load dataset
    load(testdata);
    
    for i=1:N
        if iscell(w{i})
            w{i} = cell2mat(w{i});
        end
    end
    
    repository = datarepo(w, N, V, vocabulary);
    
    %% Hyperparameters
    beta    = 0.5;    % parameter of dirichlet distribution (symmetric)
    
    a_alpha = 1; % Shape parameter of gamma prior for alpha
    b_alpha = 1; % Scale parameter of gamma prior for alpha
    alpha   = 1;

    a_gam   = 1; % Shape parameter of gamma prior for gamma
    b_gam   = 0.1; % Scale parameter of gamma prior for gamma
    gam     = 10;
    
    %% Setup test condition
    maxitr = 100;
    L = zeros(N, maxitr);
    perplexity = zeros(maxitr, 1);
    
    %% CRF initialization
    repository = crf(repository, alpha, gam, beta, true, 1);
    pi = dirichletrnd([gam; repository.m_k]);
    theta = ltopicrnd(pi, alpha, repository.n_jk, repository.J, repository.K);
    
    %% CRF inference
    vocabulary = repository.vocabulary;
    w          = repository.w;
    V          = repository.V;
    J          = repository.J;
    n_j        = repository.n_j;
    for i=1:0
        repository = crf(repository, alpha, gam, beta, false, 20);
        
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
        
        pi = dirichletrnd([gam; m_k]);
        theta = ltopicrnd(pi, alpha, n_jk, J, K);
        phi = twordrnd(V, K, n_kv, beta);
        
        drawdist(pi, theta, phi, 3);
        [perp, loglik] = wordperp(theta, phi, w, J, K, n_j);
        perplexity(i) = perp;
        L(:,i) = loglik;
        
        figure(5);
        subplot(2,1,1);imagesc(L(:,1:i));colorbar;
        subplot(2,1,2);plot(perplexity(1:i));
        
        fprintf(' alpha = %3.3f, gamma = %3.3f, beta = %3.3f, K = %d, M = %d, perplexity = %3.3f\n\n'...
            , alpha, gam, beta, K, M, perplexity(i));
        
        fprintf('-before-\n');
        K_mean = mean_k(gam, M);
        fprintf(' E(K) = %3.3f (gam = %3.3f, K = %d)\n', K_mean, gam, K);

        M_mean = mean_m(alpha, n_j);
        fprintf(' E(M) = %3.3f (alpha = %3.3f, M = %d)\n', M_mean, alpha, M);

        [gam, alpha] = hyprprmrnd(K, M, gam, alpha, n_j, a_gam, b_gam, a_alpha, b_alpha, 50);

        fprintf('-after-\n');
        K_mean = mean_k(gam, M);
        fprintf(' E(K) = %3.3f (gam = %3.3f, K = %d)\n', K_mean, gam, K);

        M_mean = mean_m(alpha, n_j);
        fprintf(' E(M) = %3.3f (alpha = %3.3f, M = %d)\n', M_mean, alpha, M);
    end
    
    %% Direct assignment inference initialization
    %init = true;
    %[topicMat, K, n_k, n_kv, n_jk, pi, theta] = directassignment(N, 0, V, w, topicMat, n_j, [], [], [], alpha, beta, gam, 1, ones(N,1), init);
    init = false;
    
   %% Direct assignment scheme inference
   diary on;
   J = repository.J;
   n_j = repository.n_j;
    for itr=1:maxitr
        totalTime = tic;
        fprintf('\n--Iteration %d--\n', itr);
        
       %% Sampling of topic
        tsTime = tic;
        fprintf(' Sampling of topic');
        diary off;
        [repository, pi, ~] = directassignment(repository, alpha, beta, gam, pi, theta, init);
        diary on;
        t = toc(tsTime);
        fprintf(' (%3.3f sec)\n', t);
        
        %% Load data
        K = repository.K;
        n_jk = repository.n_jk;
        n_kv = repository.n_kv;
        
        %% Sampling of document-topic distribution
        dtTime = tic;
        fprintf(' Sampling of document-topic distribution');
        theta = ltopicrnd(pi, alpha, n_jk, J, K);
        t = toc(dtTime);
        fprintf(' (%3.3f sec)\n', t);
        
        %% Sampling of global distribution
        gsTime = tic;
        fprintf(' Sampling of global distribution');
        
        alpha_vec = zeros(K + 1, 1);
        alpha_vec(1) = gam;
        M = 0;
        diary off;
        for k=1:K
            str = sprintf(' (%d/%d)', k, K);
            strlen = length(str);
            fprintf(str);

            M_k = 0;
            
            for d=1:N
                [M_dk, ~] = crp(n_jk(k,d), alpha*pi(k+1));
                M_k = M_k + M_dk;
            end

            alpha_vec(k+1) = M_k;
            M = M + M_k;
            
            str = repmat('\b', 1, strlen);
            fprintf(str);
        end
        diary on;
        pi = dirichletrnd(alpha_vec);
        
        t = toc(gsTime);
        fprintf(' (%3.3f sec)\n', t);
        
        %% Sample word-topic distribution
        fprintf(' Estimate word-topic distribution');
        wtTime = tic;
        phi = twordrnd(V, K, n_kv, beta);
        t = toc(wtTime);
        fprintf(' (%3.3f sec)\n', t);
        
        %% Sampling of hyper-parameters
        fprintf('-before-\n');
        K_mean = mean_k(gam, M);
        fprintf(' E(K) = %3.3f (gam = %3.3f, K = %d)\n', K_mean, gam, K);

        M_mean = mean_m(alpha, n_j);
        fprintf(' E(M) = %3.3f (alpha = %3.3f, M = %d)\n', M_mean, alpha, M);

        [gam, alpha] = hyprprmrnd(K, M, gam, alpha, n_j, a_gam, b_gam, a_alpha, b_alpha, 0);

        fprintf('-after-\n');
        K_mean = mean_k(gam, M);
        fprintf(' E(K) = %3.3f (gam = %3.3f, K = %d)\n', K_mean, gam, K);

        M_mean = mean_m(alpha, n_j);
        fprintf(' E(M) = %3.3f (alpha = %3.3f, M = %d)\n', M_mean, alpha, M);
        
        %% Estimate Dirichlet beta
        %beta = fixeditr(N_kv, N_k, beta, V);
        
        %% Log-likelihood & perplexity
        [perp, loglik] = wordperp(theta, phi, w, J, K, n_j);
        perplexity(itr) = perp;
        L(:,itr) = loglik;
        
        figure(5);
        subplot(2,1,1);imagesc(L(:,1:itr));colorbar;
        subplot(2,1,2);plot(perplexity(1:itr));
        
        %% Summary
        drawdist(pi, theta, phi, 3)
        %printTopics(N, K, theta, phi, vocabulary)
        
        t = toc(totalTime);
        fprintf(' Total elapsed time %3.3f\n', t);
        fprintf(' alpha = %3.3f, gamma = %3.3f, beta = %3.3f, K = %d, M = %d, perplexity = %3.3f\n'...
            , alpha, gam, beta, K, M, perplexity(itr));
    end
    
    printTopics(N, K, theta, phi, vocabulary)
    
    fprintf('Finished\n');
    diary off;
end

function printTopics(J, K, theta, phi, vocabulary)
    for j=1:J        
        fprintf(' Top 10 frequent words of each topic in document No.%d\n', j);

        [p, topic_idx] = sort(theta(2:K+1,j), 'descend');

        topidx = find(p > 5e-2);
        topidx_l = length(topidx);
        buf = cell(1, topidx_l);

        for k=1:topidx_l
            [A, idx] = sort(phi(:,topic_idx(k)), 'descend');
            buf{k} = [vocabulary(idx(1:10)) num2cell(A(1:10))];
        end
        disp([buf{:}]);
    end
end

function K_mean = mean_k(gam, M)
    K_mean = expcrp(gam, M);
end

function M_mean = mean_m(alpha, n_j)
    M_mean = 0;
    J = length(n_j);
    
    for j=1:J
        M_mean = M_mean + expcrp(alpha, n_j(j));
    end
end

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

function beta = fixeditr(n_kv, n_k, beta, V)
    K = length(n_k);
    KV = K*V;

    %hist = zeros(100, 1);
    
    for i=1:100
        num = sum(sum(psi(n_kv + beta))) - KV*psi(beta);
        den = V*sum(psi(n_k + beta*V)) - KV*psi(beta*V);
        
        beta_new = beta * num / den;
        
        %hist(i) = beta_new - beta;
        
        beta = beta_new;
    end
end

function x = dirichletrnd(alpha)
    x = gamrnd(alpha,1);
    x = x./sum(x);
end

function [K, p] = crp(N, alpha)
    K = 0;
    
    K = K + 1;
    count_n = zeros(K+1, 1);
    count_n(1) = 1;
    count_n(2) = alpha;
    
    for n=2:N
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
    
    p = count_n./(N+alpha);
    p = p./sum(p);
end

function drawdist(pi, theta, phi, fignum)
    figure(fignum); 
    subplot(1,12,1); imagesc(pi); caxis([0 1]); set(gca, 'XTick', []); title('global-level');
    subplot(1,12,3:12); imagesc(theta); caxis([0 1]); title('local-level');
    
    figure(fignum+1); imagesc(phi'); caxis([0 1]); title('topic-word distribution');
    
    drawnow();
end

function [perp, L] = wordperp(theta, phi, w, J, K, n_j)
    L = zeros(J, 1);

    for j=1:J
        for n=1:n_j(j)
            L_buf = 0;
            for k=1:K
                L_buf = L_buf + theta(k+1,j) * phi(w{j}(n),k);
            end
            L(j) = L(j) + log(L_buf);
        end
    end

    perp = exp(-sum(L)/sum(n_j));
end

function repository = crf(repository, alpha, gam, beta, init, maxitr)
    if ~exist('maxitr','var')
        maxitr = 100;
    end
    
    %% load data
    w = repository.w;
    topicMat = repository.topicMat;
    tableMat = repository.tableMat;
    phi = repository.phi;
    
    V = repository.V;
    J = repository.J;
    K = repository.K;
    M = repository.M;
    
    n_j   = repository.n_j;
    n_k   = repository.n_k;
    n_kv  = repository.n_kv;
    n_jt  = repository.n_jt;
    n_jtv = repository.n_jtv;
    
    m_k   = repository.m_k;
        
    %% Calculate constants
    betaV = beta*V;
    loggam = log(gam);
    logalpha = log(alpha);
    logV = log(V);
    loggam_betaV = loggamfun(betaV);
    Vloggam_beta = V*log(gamma((beta)));
    
    %% Main loop
    for itr=1:maxitr
        fprintf('iteration: %d\n', itr);
        
       %% Sampling tables
        M_ini = M;
        K_ini = K;
        tspawned = 0;
        tremoved = 0;
        kspawned = 0;
        kremoved = 0;
        
        for j=1:J
            str = sprintf(' (%d/%d)', j, J);
            strlen = length(str);
            fprintf(str);
            
            T = length(n_jt{j});
            logp = zeros(T + 1, 1);
            
            for n=1:n_j(j)
                v = w{j}(n);
                
                if ~init
                    t = tableMat{j}(n);
                    k = topicMat{j}(n);
                    
                    assert(phi{j}(t) == topicMat{j}(n), 'CRF: topic assignment is inconsistent');
                    assert(isequal(n_jt{j}, sum(n_jtv{j},2)), 'CRF: n_jt and n_jtv is incosistent');
                    assert(n_jtv{j}(t,v) > 0, 'CRF: table assignment is inconsistent');
                    
                    n_jt{j}(t) = n_jt{j}(t) - 1;
                    n_kv(k,v) = n_kv(k,v) - 1;
                    n_k(k) = n_k(k) - 1;
                    n_jtv{j}(t,v) = n_jtv{j}(t,v) - 1;
                    
                    if n_jt{j}(t) == 0
                        tremoved = tremoved + 1;
                        
                        M = M - 1;
                        
                        T = length(n_jt{j}) - 1;
                        
                        n_jt_new = zeros(T, 1);
                        n_jt_new(1:t-1) = n_jt{j}(1:t-1);
                        n_jt_new(t:T) = n_jt{j}(t+1:T+1);
                        n_jt{j} = n_jt_new;
                        
                        phi_new = zeros(T, 1);
                        phi_new(1:t-1) = phi{j}(1:t-1);
                        phi_new(t:T) = phi{j}(t+1:T+1);
                        phi{j} = phi_new;
                        
                        n_jtv_new = zeros(T, V);
                        n_jtv_new(1:t-1,:) = n_jtv{j}(1:t-1,:);
                        n_jtv_new(t:T,:) = n_jtv{j}(t+1:T+1,:);
                        n_jtv{j} = n_jtv_new;
                        
                        m_k(k) = m_k(k) - 1;
                        
                        idx_t = find(tableMat{j} > t);
                        tableMat{j}(idx_t) = tableMat{j}(idx_t) - 1;
                        
                        if n_k(k) == 0
                            assert(m_k(k) == 0, 'CRF: empty n_k and m_k is inconsistent (table)');
                            
                            kremoved = kremoved + 1;
                            K = K - 1;
                            
                            [n_k, n_kv, m_k, topicMat, phi] = deleteTopics(k, K, V, J, n_k, n_kv, m_k, topicMat, phi);
                        end
                        
                        logp = zeros(T + 1, 1);
                    end
                end
                
                for t=1:T
                    k = phi{j}(t);
                    logp_buf = log(n_jt{j}(t)) + log((n_kv(k,v) + beta)) - log((n_k(k) + betaV));
                    
                    logp(t) = logp_buf;
                end

                p_buf = sum(m_k ./ (M + gam) .* (n_kv(:,v) + beta) ./ (n_k + betaV))...
                    + gam / (M + gam) * (1/V);
                logp_buf = logalpha + log(p_buf);
                logp(T+1) = logp_buf;
                
                idx = logpmnrnd(logp);
                
                if idx <= T
                    t = idx;
                    k = phi{j}(t);
                    
                    n_kv(k,v) = n_kv(k,v) + 1;
                    n_k(k) = n_k(k) + 1;
                    n_jt{j}(t) = n_jt{j}(t) + 1;
                    n_jtv{j}(t,v) = n_jtv{j}(t,v) + 1;
                elseif idx == (T + 1)
                    logp = zeros(K + 1, 1);
                    
                    for k=1:K
                        logp_buf = log(m_k(k)) + log(n_kv(k,v) + beta) - log(n_k(k) + betaV);
                        logp(k) = logp_buf;
                    end

                    logp(K + 1) = loggam - logV;
                    
                    idx = logpmnrnd(logp);
                    
                    if idx <= K
                        tspawned = tspawned + 1;
                    
                        M = M + 1;

                        k = idx;

                        n_kv(k,v) = n_kv(k,v) + 1;
                        n_k(k) = n_k(k) + 1;
                        m_k(k) = m_k(k) + 1;
                        n_jt{j} = [n_jt{j}; 1];

                        T = length(n_jt{j});
                        logp = zeros(T + 1, 1);
                        phi{j} = [phi{j}; k];

                        t = T;

                        n_jtv{j} = [n_jtv{j}; zeros(1,V)];
                        n_jtv{j}(t,v) = 1;
                    elseif idx == (K + 1)
                        tspawned = tspawned + 1;
                        kspawned = kspawned + 1;

                        M = M + 1;
                        K = K + 1;

                        k = K;

                        n_kv = [n_kv; zeros(1,V)];
                        n_kv(k,v) = 1;
                        n_k = [n_k; 1];
                        m_k = [m_k; 1];
                        n_jt{j} = [n_jt{j}; 1];

                        T = length(n_jt{j});
                        logp = zeros(T + 1, 1);
                        phi{j} = [phi{j}; k];

                        t = T;

                        n_jtv{j} = [n_jtv{j}; zeros(1,V)];
                        n_jtv{j}(t,v) = 1;
                    else
                        assert(false, 'CRF: indexing is not working (new table)');
                    end
                else
                    assert(false, 'CRF: indexing is not working (existing table)');
                end
                
                topicMat{j}(n) = k;
                tableMat{j}(n) = t;
            end
            
            assert(K == length(m_k) && K == length(n_k), 'CRF: K and length is inconsistent');
            assert(isequal(n_k, sum(n_kv,2)), 'CRF: n_k and n_kv is inconsistent');
            assert(M == sum(m_k) && M == sum(cell2mat(cellfun(@(x) length(x), n_jt, 'UniformOutput', false)))...
                && M == sum(cell2mat(cellfun(@(x) length(x), phi, 'UniformOutput', false)))...
                ,'CRF: M and sum is inconsistent');
            
            str = repmat('\b', 1, strlen);
            fprintf(str);
        end
        
        assert(sum(n_j) == sum(n_k) && sum(n_j) == sum(cell2mat(cellfun(@(x) sum(x), n_jt, 'UniformOutput', false)))...
             ,'CRF: n_j, n_k and n_jt is incosistent');
        
        figure(1);
        subplot(211); stem(m_k, 'Marker', 'None'); title('m_k');
        subplot(212); stem(n_k, 'Marker', 'None'); title('n_k');
        
        figure(2);
        imagesc(n_kv); caxis([0 max(max(n_kv))]); set(gca, 'XTick', []); title('n_k_v');
        
        drawnow();
        
        assert(M == (M_ini + tspawned - tremoved), 'add/remove count of M is inoconsistent');
        assert(K == (K_ini + kspawned - kremoved), 'add/remove count of K is inoconsistent');
        fprintf('table_num_update: %d = %d + %d - %d\n', M, M_ini, tspawned, tremoved);
        fprintf('topic_num_update: %d = %d + %d - %d\n', K, K_ini, kspawned, kremoved);
        
        init = false;
        
       %% Sampling topics
        K_ini = K;
        kspawned = 0;
        kremoved = 0;
        
        for j=1:J
            str = sprintf(' (%d/%d)', j, J);
            strlen = length(str);
            fprintf(str);
            
            T = length(n_jt{j});
            logp = zeros(K + 1, 1);
            
            assert(isequal(n_jt{j}, sum(n_jtv{j},2)), 'CRF: n_jt and n_jtv is incosistent');
            
            for t=1:T    
                % remove target component
                k = phi{j}(t);
                m_k(k) = m_k(k) - 1;
                n_k(k) = n_k(k) - n_jt{j}(t);
                n_kv(k,:) = n_kv(k,:) - n_jtv{j}(t,:);
                
                if m_k(k) == 0
                    assert(n_k(k) == 0, 'CRF: empty n_k and m_k is inconsistent (topic)');
                    kremoved = kremoved + 1;
                    K = K -1;
                    
                    [n_k, n_kv, m_k, topicMat, phi] = deleteTopics(k, K, V, J, n_k, n_kv, m_k, topicMat, phi);
                    logp = zeros(K + 1, 1);
                end
                
                for k=1:K
                    logp_buf...
                        = log(m_k(k)) + loggamfun(n_k(k) + betaV) - loggamfun(n_k(k) + n_jt{j}(t) + betaV)...
                        + sum(loggamfun(n_kv(k,:) + n_jtv{j}(t,:) + beta)) - sum(loggamfun(n_kv(k,:) + beta));
                    
                    logp(k) = logp_buf;
                end
                
                logp_buf...
                    = loggam + loggam_betaV - loggamfun(n_jt{j}(t) + betaV)...
                    + sum(loggamfun(n_jtv{j}(t,:) + beta)) - Vloggam_beta;
                logp(K+1) = logp_buf;
                
                idx = logpmnrnd(logp);
                
                if idx <= K
                    k = idx;
                    
                    phi{j}(t) = k;
                    m_k(k) = m_k(k) + 1;
                    n_k(k) = n_k(k) + n_jt{j}(t);
                    n_kv(k,:) = n_kv(k,:) + n_jtv{j}(t,:);
                elseif idx == (K+1)
                    kspawned = kspawned + 1;
                    
                    K = K + 1; 
                    
                    k = K;
                    
                    phi{j}(t) = k;
                    m_k = [m_k; 1];
                    n_k = [n_k; n_jt{j}(t)];
                    n_kv = [n_kv; n_jtv{j}(t,:)];
                    
                    logp = zeros(K + 1, 1);
                else
                    assert(false, 'CRF: indexing is not working (topic)');
                end
                
                t_idx = tableMat{j} == t;
                topicMat{j}(t_idx) = k;
            end
            
            assert(K == length(m_k) && K == length(n_k), 'CRF: K and length is inconsistent');
            assert(isequal(n_k, sum(n_kv,2)), 'CRF: n_k and n_kv is inconsistent');
            assert(M == sum(m_k) && M == sum(cell2mat(cellfun(@(x) length(x), n_jt, 'UniformOutput', false)))...
                && M == sum(cell2mat(cellfun(@(x) length(x), phi, 'UniformOutput', false)))...
                ,'CRF: M and sum is inconsistent');
            
            str = repmat('\b', 1, strlen);
            fprintf(str);
        end
        
        assert(sum(n_j) == sum(n_k) && sum(n_j) == sum(cell2mat(cellfun(@(x) sum(x), n_jt, 'UniformOutput', false)))...
             ,'CRF: n_j, n_k and n_jt is incosistent');
        assert(isequal(n_jt{j}, sum(n_jtv{j},2)), 'CRF: n_jt and n_jtv is incosistent');
        
        figure(1);
        subplot(211); stem(m_k, 'Marker', 'None'); title('m_k');
        subplot(212); stem(n_k, 'Marker', 'None'); title('n_k');
        
        figure(2);
        imagesc(n_kv); caxis([0 max(max(n_kv))]); set(gca, 'XTick', []); title('n_k_v');
        
        drawnow();
        
        assert(K == (K_ini + kspawned - kremoved), 'add/remove count of K is inoconsistent');
        fprintf('topic_num_update: %d = %d + %d - %d\n\n', K, K_ini, kspawned, kremoved);
    end
    
    %% Set to repository (return: topicMat, tableMat, K, M, n_k, n_kv, m_k, n_j)
    repository.topicMat = topicMat;
    repository.tableMat = tableMat;
    repository.phi = phi;
    
    repository.K = K;
    repository.M = M;
    
    repository.n_k   = n_k;
    repository.n_kv  = n_kv;
    repository.n_jt  = n_jt;
    repository.n_jtv = n_jtv;
    
    repository.m_k   = m_k;
    
    repository.n_jk  = repository.count_jk();
end

function [repository, pi, theta] = directassignment(repository, alpha, beta, gam, pi, theta, init)
    if ~exist('init','var')
        init = false;
    end
    
    %% load data
    J = repository.J;
    K = repository.K;
    V = repository.V;
    
    w = repository.w;
    topicMat = repository.topicMat;
    
    n_j  = repository.n_j;
    n_k  = repository.n_k;
    n_kv = repository.n_kv;
    n_jk = repository.n_jk;
    
    %% Calculate constants
    logV = log(V);
    betaV = beta*V;

    docidx = randperm(J);
    logp = zeros(K + 1, 1);

    for j=1:J
        str = sprintf(' (%d/%d)', j, J);
        strlen = length(str);
        fprintf(str);

        d_idx = docidx(j);
        W = n_j(d_idx);

        for n=1:W
            v = w{d_idx}(n);
            
            if ~init
                z_dn = topicMat{d_idx}(n);

                n_k(z_dn, 1) = n_k(z_dn, 1) - 1;
                n_kv(z_dn, v) = n_kv(z_dn, v) - 1;
                n_jk(z_dn, d_idx) = n_jk(z_dn, d_idx) - 1;

                % Pack empty topic
                if n_k(z_dn) == 0
                    pi_new = zeros(K, 1);
                    pi_new(1:z_dn) = pi(1:z_dn);
                    pi_new(z_dn+1:K) = pi(z_dn+2:K+1);
                    pi_new(1) = pi_new(1) + pi(z_dn+1);
                    pi = pi_new;

                    theta_new = zeros(K, J);
                    theta_new(1:z_dn,:) = theta(1:z_dn,:);
                    theta_new(z_dn+1:K,:) = theta(z_dn+2:K+1,:);
                    theta_new(1,:) = theta_new(1,:) + theta(z_dn+1,:);
                    theta = theta_new;

                    K = K - 1;

                    n_k_new = zeros(K, 1);
                    n_k_new(1:z_dn-1) = n_k(1:z_dn-1);
                    n_k_new(z_dn:K) = n_k(z_dn+1:K+1);
                    n_k = n_k_new;

                    n_kv_new = zeros(K, V);
                    n_kv_new(1:z_dn-1, :) = n_kv(1:z_dn-1, :);
                    n_kv_new(z_dn:K, :) = n_kv(z_dn+1:K+1, :);
                    n_kv = n_kv_new;

                    n_jk_new = zeros(K, J);
                    n_jk_new(1:z_dn-1,:) = n_jk(1:z_dn-1,:);
                    n_jk_new(z_dn:K,:) = n_jk(z_dn+1:K+1,:);
                    n_jk = n_jk_new;

                    idx = cellfun(@(x) find(x > z_dn), topicMat, 'UniformOutput', false);
                    for i=1:J
                        topicMat{i}(idx{i}) = topicMat{i}(idx{i}) - 1;
                    end

                    logp = zeros(K + 1, 1);
                end
            end

            for k=1:K
                log_f_k = log(n_kv(k,v) + beta) - log(n_k(k,1) + betaV);
                tmp = n_jk(k,d_idx) + alpha*pi(k+1);

                logp(k) = log(tmp) + log_f_k;
            end
            tmp = alpha*pi(1);
            logp(K+1) = log(tmp) - logV;

            z = logpmnrnd(logp);

            % spawn new topic
            if z == (K+1)
                K = K + 1;

                mu_0 = betarnd(1, gam);
                pi_new_0 = pi(1) * (1 - mu_0);
                pi_new_K = pi(1) * mu_0;

                pi_new = zeros(K+1,1);
                pi_new(1) = pi_new_0;
                pi_new(2:K) = pi(2:K);
                pi_new(K+1) = pi_new_K;

                mu_d = betarnd(alpha*pi(1)*mu_0, alpha*pi(1)*(1 - mu_0), 1, J);
                theta_new_0 = theta(1,:).*(1 - mu_d);
                theta_new_K = theta(1,:).*mu_d;

                theta_new = zeros(K+1,J);
                theta_new(1,:) = theta_new_0;
                theta_new(2:K,:) = theta(2:K,:);
                theta_new(K+1,:) = theta_new_K;

                pi = pi_new;
                theta = theta_new;

                logp = zeros(K + 1, 1);

                n_k = [n_k; 0];
                n_kv = [n_kv; zeros(1, V)];
                n_jk = [n_jk; zeros(1, J)];
            end

            topicMat{d_idx}(n) = z;
            n_k(z, 1) = n_k(z, 1) + 1;
            n_kv(z, v) = n_kv(z, v) + 1;
            n_jk(z, d_idx) = n_jk(z, d_idx) + 1;
        end

        str = repmat('\b', 1, strlen);
        fprintf(str);
    end
    
    %% Validation
    assert(sum(n_j) == sum(n_k), 'n_k count is corrupted');
    assert(sum(n_j) == sum(sum(n_kv)), 'n_kv count is corrupted');
    assert(sum(n_j) == sum(sum(n_jk)), 'n_jk count is corrupted');

    assert(isempty(find(n_k < 1, 1)), 'n_k contains minus');
    assert(isempty(find(n_kv < 0, 1)), 'n_kv contains minus');
    assert(isempty(find(n_jk < 0, 1)), 'n_jk contains minus');
        
    A = cellfun(@(x) find(x < 1), topicMat, 'UniformOutput', false);
    cellfun(@(x) assert(isempty(x), 'null topic is contained'), A, 'UniformOutput', false);
        
    %% Set to repository
    repository.K        = K;
    repository.topicMat = topicMat;
    repository.n_k      = n_k;
    repository.n_kv     = n_kv;
    repository.n_jk     = n_jk;
end

function idx = logpmnrnd(logp)
    p = exp(logp - max(logp));
    p = p./sum(p);
    idx = mnrnd(1, p);
    idx = find(idx == 1);
end

%% deletion of topics
function [n_k, n_kv, m_k, topicMat, phi] = deleteTopics(k, K, V, J, n_k, n_kv, m_k, topicMat, phi)
    n_k_new = zeros(K, 1);
    n_k_new(1:k-1) = n_k(1:k-1);
    n_k_new(k:K) = n_k(k+1:K+1);
    n_k = n_k_new;

    n_kv_new = zeros(K, V);
    n_kv_new(1:k-1,:) = n_kv(1:k-1,:);
    n_kv_new(k:K,:) = n_kv(k+1:K+1,:);
    n_kv = n_kv_new;

    m_k_new = zeros(K, 1);
    m_k_new(1:k-1) = m_k(1:k-1);
    m_k_new(k:K) = m_k(k+1:K+1);
    m_k = m_k_new;

    idx_k = cellfun(@(x) find(x > k), topicMat, 'UniformOutput', false);
    idx_p = cellfun(@(x) find(x > k), phi, 'UniformOutput', false);
    for i=1:J
        topicMat{i}(idx_k{i}) = topicMat{i}(idx_k{i}) - 1;
        phi{i}(idx_p{i}) = phi{i}(idx_p{i}) - 1;
    end
end

%% topic-word distribution
function phi = twordrnd(V, K, n_kv, beta)
    phi = zeros(V, K);
    
    for k=1:K
        phi(:,k) = dirichletrnd(n_kv(k, :) + beta);
    end
end

%% document-topic distribution
function theta = ltopicrnd(pi, alpha, n_jk, J, K)
    theta = zeros(K+1, J);

    for j=1:J
        alpha_vec = pi.*alpha;
        alpha_vec(2:K+1,1) = alpha_vec(2:K+1,1) + n_jk(:,j);
        
        theta(:,j) = dirichletrnd(alpha_vec);
    end
end

function K = count_k(topicMat)
    J = length(topicMat);

    uniqueT = cellfun(@(x) unique(x), topicMat, 'UniformOutput', false);
    numelem = cellfun(@(x) length(x), uniqueT, 'UniformOutput', false);
    numelemsum = sum([numelem{:}]);
    k_array = zeros(numelemsum,1);
    
    counter = 1;
    
    for d=1:J
        for k=1:numelem{d}
            k_array(counter) = uniqueT{d}(k);
            counter = counter + 1;
        end
    end
    
    K = length(unique(k_array));
end

%% count n_kv
function [n_kv] = count_kv(topicMat, w, K, V)
    n_kv = zeros(K, V);
    
    for k=1:K
        topic_idx = cellfun(@(x) x == k, topicMat, 'UniformOutput', false);
        topic_idx = cellfun(@(x) x(:), topic_idx, 'UniformOutput', false);

        fprintf('K = %d/%d, N_kv: ', k, K);
        for v=1:V
            msg = sprintf('%d/%d', v, V);
            fprintf(msg);
            
            word_idx = cellfun(@(x) x == v, w, 'UniformOutput', false);
            word_idx = cellfun(@(x) x(:), word_idx, 'UniformOutput', false);
            numk = cellfun(@(x,y) length(find(x & y)), word_idx, topic_idx, 'UniformOutput', false);
            
            n_kv(k,v) = sum([numk{:}]);
            
            fprintf(repmat('\b',1,length(msg)));
        end

        fprintf('...\n');
    end
end

function N = expcrp(alpha, n)
    N = alpha * (psi(alpha + n) - psi(alpha));
end

%% approximation of log-gamma function
% For example, compare result of the calculation below.
% A = loggamaprx(50);
% B = log(gamma(50));
function y = loggamfun(x)
    y1 = log(gamma(x));
    idx = isinf(y1);
    
    x2 = x(idx);
    y2 = 0.5.*(log(2*pi) - log(x2)) + x2.*(log(x2 + 1./(12*x2 - 1./(10*x2))) - 1);
    
    y = y1;
    y(idx) = y2;
end