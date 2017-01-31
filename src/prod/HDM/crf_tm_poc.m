function crf_tm_poc(testdata)
    %% Setup
    logfilename = strcat('result_rev5_', sprintf('%5.3f',now),'.txt'); 
    homedir = userpath;
    homedir = homedir(1:(length(homedir) - 1));
    logging = strcat(homedir,'/logs/HDM/',logfilename);
    diary(logging);
    fprintf('*** Experiment %5.5f ***\n', now);
    
    %% Test setting
    maxitr = 100; % Number of iterations for inference
    
    load(testdata);
    L = zeros(N, maxitr);
    perplexity = zeros(maxitr, 1);
    
    beta = 0.001;    % parameter of dirichlet distribution (symmetric)
    
    a_alpha = 1; % Shape parameter of gamma prior for alpha
    b_alpha = 1; % Scale parameter of gamma prior for alpha
    %alpha = gamrnd(a_alpha, 1/b_alpha); % Concentration parameter of children distribution
    alpha = 1;

    a_gam = 1; % Shape parameter of gamma prior for gamma
    b_gam = 0.1; % Scale parameter of gamma prior for gamma
    %gam = gamrnd(a_gam, 1/b_gam);  % Concentration parameter of parent distribution
    gam = 10;
    
    for i=1:N
        if iscell(w{i})
            w{i} = cell2mat(w{i});
        end
    end
    
    for i=1:1
        [m_k, n_k, n_kv, n_j, K, M, topicMat] = crf(w, V, N, alpha, gam, beta, 20);
        
        [n_jk, n_k_check] = count_jk(topicMat, N, K);
        assert(isequal(n_k, n_k_check), 'n_k count is inconsistent');
        
        if mod(i,20) == 0
            K_check = count_k(topicMat);
            assert(K == K_check, 'K count is inconsistent');
            
            [n_kv_check] = count_kv(topicMat, w, K, V);
            assert(isequal(n_kv', n_kv_check), 'n_kv count is inconsistent');
        end
        
        pi = dirichletrnd([gam; m_k]);
        theta = ltopicrnd(pi, alpha, n_jk, N, K);
        phi = twordrnd(V, K, n_kv', beta);
        
        %{
        fprintf('-before-\n');
        [K_mean] = mean_k(gam, M);
        fprintf(' K = %d (gam = %3.3f: %3.3f)\n', K, gam, K_mean);

        M_mean = mean_m(alpha, n_j);
        fprintf(' M = %d (alpha = %3.3f: %3.3f)\n', M, alpha, M_mean);
        
        [gam, alpha] = hyprprmrnd(K, M, gam, alpha, n_j, a_gam, b_gam, a_alpha, b_alpha);
        
        fprintf('-after-\n');
        [K_mean] = mean_k(gam, M);
        fprintf(' K = %d (gam = %3.3f: %3.3f)\n', K, gam, K_mean);

        M_mean = mean_m(alpha, n_j);
        fprintf(' M = %d (alpha = %3.3f: %3.3f)\n', M, alpha, M_mean);
        %}
        
        drawdist(pi, theta, phi, 3);
        [perp, loglik] = wordperp(theta, phi, w, N, K, n_j);
        perplexity(i) = perp;
        L(:,i) = loglik;
        
        figure(5);
        subplot(2,1,1);imagesc(L(:,1:i));colorbar;
        subplot(2,1,2);plot(perplexity(1:i));
        
        printTopics(N, K, theta, phi, vocabulary);
        
        fprintf(' alpha = %3.3f, gamma = %3.3f, beta = %3.3f, K = %d, perplexity = %3.3f\n'...
            , alpha, gam, beta, K, perplexity(i));
    end
    
    %% Count items
    diary off
    n_j = cell2mat(cellfun(@(x) length(x), w, 'UniformOutput', false));
    K = count_k(topicMat);
    [N_dk, N_k] = count_jk(topicMat, N, K);
    [N_kv] = count_kv(topicMat, w, K, V);
    diary on;
    
    if ~exist('theta','var')
        theta = rand(K+1,N);
        theta = theta./repmat(sum(theta,1), K+1, 1);
    end
    if ~exist('pi','var')
        pi = rand(K+1,1);
        pi = pi./sum(pi);
    end
        
    %% Sampling from posterior
    for itr=1:maxitr
        totalTime = tic;
        fprintf('--Iteration %d--\n', itr);
        
        %% Direct assignment scheme
        % Sampling of topic
        tsTime = tic;
        fprintf(' Sampling of topic');
        docidx = randperm(N);
        
        p = zeros(1+K+1, 1); %p(1) = 0, p(2) = k_1, p(3) = k_2, ..., p(K+1) = k_K, p(K+2) = k_0
                
        diary off;
        for d=1:N
            str = sprintf('\n (%d/%d)', d, N);
            strlen = length(str);
            fprintf(str);
            
            d_idx = docidx(d);
            W = n_j(d_idx);
            r = rand(W,1);
            
            for n=1:W
                v = w{d_idx}(n);
                z_dn = topicMat{d_idx}(n);
                
                N_k(z_dn, 1) = N_k(z_dn, 1) - 1;
                N_kv(v, z_dn) = N_kv(v, z_dn) - 1;
                N_dk(z_dn, d_idx) = N_dk(z_dn, d_idx) - 1;
                
                for k=1:K
                    f_k = (N_kv(v,k)+beta) / (N_k(k,1)+beta*V);
                    tmp = N_dk(k,d_idx) + alpha*pi(k+1);

                    p(k+1) = tmp * f_k + p(k);
                end
                f_k = 1/V;
                tmp = alpha*pi(1);
                p(K+2) =  tmp * f_k + p(K+1);

                p = p ./ p(end);
                
                idx = find(p < r(n));
                z = idx(end);
                
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
                    
                    mu_d = betarnd(alpha*pi(1)*mu_0, alpha*pi(1)*(1 - mu_0), 1, N);
                    theta_new_0 = theta(1,:).*(1 - mu_d);
                    theta_new_K = theta(1,:).*mu_d;
                    
                    theta_new = zeros(K+1,N);
                    theta_new(1,:) = theta_new_0;
                    theta_new(2:K,:) = theta(2:K,:);
                    theta_new(K+1,:) = theta_new_K;
                    
                    pi = pi_new;
                    theta = theta_new;
                    
                    p = zeros(1+K+1, 1);
                    
                    N_k = [N_k; 0];
                    N_kv = [N_kv zeros(V,1)];
                    N_dk = [N_dk; zeros(1, N)];
                end
                
                topicMat{d_idx}(n) = z;
                N_k(z, 1) = N_k(z, 1) + 1;
                N_kv(v, z) = N_kv(v, z) + 1;
                N_dk(z, d_idx) = N_dk(z, d_idx) + 1;
                
                % Pack empty topic
                if N_k(z_dn) == 0
                    pi_new = zeros(K, 1);
                    pi_new(1:z_dn) = pi(1:z_dn);
                    pi_new(z_dn+1:K) = pi(z_dn+2:K+1);
                    pi_new(1) = pi_new(1) + pi(z_dn+1);
                    pi = pi_new;
                    
                    theta_new = zeros(K, N);
                    theta_new(1:z_dn,:) = theta(1:z_dn,:);
                    theta_new(z_dn+1:K,:) = theta(z_dn+2:K+1,:);
                    theta_new(1,:) = theta_new(1,:) + theta(z_dn+1,:);
                    theta = theta_new;
                    
                    K = K - 1;
                    
                    N_k_new = zeros(K, 1);
                    N_k_new(1:z_dn-1) = N_k(1:z_dn-1);
                    N_k_new(z_dn:K) = N_k(z_dn+1:K+1);
                    N_k = N_k_new;
                    
                    N_kv_new = zeros(V, K);
                    N_kv_new(:,1:z_dn-1) = N_kv(:,1:z_dn-1);
                    N_kv_new(:,z_dn:K) = N_kv(:,z_dn+1:K+1);
                    N_kv = N_kv_new;
                    
                    N_dk_new = zeros(K, N);
                    N_dk_new(1:z_dn-1,:) = N_dk(1:z_dn-1,:);
                    N_dk_new(z_dn:K,:) = N_dk(z_dn+1:K+1,:);
                    N_dk = N_dk_new;
                    
                    idx = cellfun(@(x) find(x > z_dn), topicMat, 'UniformOutput', false);
                    for i=1:N
                        topicMat{i}(idx{i}) = topicMat{i}(idx{i}) - 1;
                    end
                    
                    p = zeros(1+K+1, 1);
                end
            end
            
            str = repmat('\b', 1, strlen);
            fprintf(str);
        end
        diary on;
        t = toc(tsTime);
        fprintf(' (%3.3f sec)\n', t);
        
        assert(sum(n_j) == sum(N_k), 'N_k count is corrupted');
        assert(sum(n_j) == sum(sum(N_kv)), 'N_kv count is corrupted');
        assert(sum(n_j) == sum(sum(N_dk)), 'N_dk count is corrupted');
        
        assert(isempty(find(N_k < 1, 1)), 'N_k contains minus');
        assert(isempty(find(N_kv < 0, 1)), 'N_kv contains minus');
        assert(isempty(find(N_dk < 0, 1)), 'N_dk contains minus');
        
        A = cellfun(@(x) find(x < 1), topicMat, 'UniformOutput', false);
        cellfun(@(x) assert(isempty(x), 'null topic is contained'), A, 'UniformOutput', false);
        
        % Sampling of document-topic distribution
        dtTime = tic;
        fprintf(' Sampling of document-topic distribution');
        theta = ltopicrnd(pi, alpha, N_dk, N, K);
        t = toc(dtTime);
        fprintf(' (%3.3f sec)\n', t);
        
        % Sampling of global distribution
        gsTime = tic;
        fprintf(' Sampling of global distribution');
        
        alpha_vec = zeros(K+1,1);
        alpha_vec(1) = gam;
        M = 0;
        for k=1:K
            M_k = 0;
            
            for d=1:N
                [M_dk, ~] = crp(N_dk(k,d), alpha*pi(k+1));
                M_k = M_k + M_dk;
            end

            alpha_vec(k+1) = M_k;
            M = M + M_k;
        end
        pi = dirichletrnd(alpha_vec);
        
        t = toc(gsTime);
        fprintf(' (%3.3f sec)\n', t);
        
        % Sample word-topic distribution
        fprintf(' Estimate word-topic distribution');
        wtTime = tic;
        phi = twordrnd(V, K, N_kv, beta);
        t = toc(wtTime);
        fprintf(' (%3.3f sec)\n', t);
        
        % Sampling of hyper-parameters
        if itr > 1000
            fprintf('-before-\n');
            [K_mean] = mean_k(gam, M);
            fprintf(' K = %d (gam = %3.3f: %3.3f)\n', K, gam, K_mean);

            M_mean = mean_m(alpha, n_j);
            fprintf(' M = %d (alpha = %3.3f: %3.3f)\n', M, alpha, M_mean);

            [gam, alpha] = hyprprmrnd(K, M, gam, alpha, n_j, a_gam, b_gam, a_alpha, b_alpha);

            fprintf('-after-\n');
            [K_mean] = mean_k(gam, M);
            fprintf(' K = %d (gam = %3.3f: %3.3f)\n', K, gam, K_mean);

            M_mean = mean_m(alpha, n_j);
            fprintf(' M = %d (alpha = %3.3f: %3.3f)\n', M, alpha, M_mean);
        end
        
        % Estimate Dirichlet beta
        %beta = fixeditr(N_kv, N_k, beta, V);
        
        % Log-likelihood & perplexity
        [perp, loglik] = wordperp(theta, phi, w, N, K, n_j);
        perplexity(itr) = perp;
        L(:,itr) = loglik;
        
        figure(1);
        subplot(2,1,1);imagesc(L(:,1:itr));colorbar;
        subplot(2,1,2);plot(perplexity(1:itr));
        
        % Summary
        drawdist(pi, theta, phi, 3)
        printTopics(N, K, theta, phi, vocabulary)
        
        t = toc(totalTime);
        fprintf(' Total elapsed time %3.3f\n', t);
        fprintf(' alpha = %3.3f, gamma = %3.3f, beta = %3.3f, K = %d, perplexity = %3.3f\n'...
            , alpha, gam, beta, K, perplexity(itr));
    end
    
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
    K_mean = 1;
    
    for i=1:M-1
        K_mean = K_mean + gam/(gam + i);
    end
end

function M_mean = mean_m(alpha, n_j)
    M_mean = 0;
    J = length(n_j);
    
    for j=1:J
        buf = 1;
        
        for i=1:n_j(j)-1
            buf = buf + alpha/(alpha + i);
        end
        
        M_mean = M_mean + buf;
    end
end

function [gam, alpha] = hyprprmrnd(K, M, gam, alpha, n_j, a_gam, b_gam, a_alpha, b_alpha)
    J = length(n_j);
    I = ones(J,1);
    
    for i=1:50
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

function beta = fixeditr(N_kv, N_k, beta, V)
    K = length(N_k);

    %hist = zeros(100, 1);
    
    for i=1:100
        num = sum(sum(psi(N_kv + beta))) - K*V*psi(beta);
        den = V*sum(psi(N_k + beta*V)) - K*V*psi(beta*V);
        
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
    subplot(1,12,1); imagesc(pi); caxis([0 1]); set(gca, 'XTick', []);
    subplot(1,12,3:12); imagesc(theta); caxis([0 1]);
    
    figure(fignum+1); imagesc(phi'); caxis([0 1]);
    
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

%phi<j,t>, t<j,i>, theta<j,i>, k<j,t>, n<j,k,t>
function [m_k, n_k, n_kv, n_j, K, M, topicMat] = crf(w, V, J, alpha, gam, beta, maxitr)
    % Setup
    if ~exist('maxitr','var')
        maxitr = 100;
    end
    
    K = 0;
    M = 0;

    topicMat = w;
    tableMat = w;
    n_j = cell2mat(cellfun(@(x) length(x), w, 'UniformOutput', false));
    
    n_k = zeros(K, 1);
    n_kv = zeros(K, V);
    m_k = zeros(K, 1);

    n_jt = cell(J, 1);
    phi = cell(J, 1);
    n_jtv = cell(J, 1);
    for j=1:J
        n_jt{j} = [];
        phi{j} = [];
        n_jtv{j} = [];
    end
        
    betaV = beta*V;
    log_gam = log(gam);
    loggam_betaV = loggamaprx(betaV);
    Vloggam_beta = V*log(gamma((beta)));
    
    for itr=1:maxitr
        fprintf('iteration: %d\n', itr);
        
        % Sampling tables
        M_ini = M;
        K_ini = K;
        tspawned = 0;
        tremoved = 0;
        kspawned = 0;
        kremoved = 0;
        
        for j=1:J
            T = length(n_jt{j});
            p = zeros(1 + T + K + 1, 1);
            
            rnd = rand(n_j(j), 1);

            for n=1:n_j(j)
                v = w{j}(n);
                
                if itr > 1
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
                            
                            [K, n_k, n_kv, m_k, topicMat, phi] = deleteTopics(k, K, V, J, n_k, n_kv, m_k, topicMat, phi);
                        end
                        
                        p = zeros(1 + K + T + 1, 1);
                    end
                end
                
                cnt = 1;

                for t=1:T
                    k = phi{j}(t);
                    p(cnt + 1) = n_jt{j}(t) * (n_kv(k,v) + beta) / (n_k(k) + betaV) + p(cnt);
                    cnt = cnt + 1;
                end

                for k=1:K
                    p(cnt + 1) = alpha * m_k(k) / (M + gam) * (n_kv(k,v) + beta) / (n_k(k) + betaV) + p(cnt);
                    cnt = cnt + 1;
                end

                p(cnt + 1) = alpha * gam / (M + gam) / V + p(cnt);

                idx = find(p < p(end)*rnd(n));
                idx = idx(end);

                if idx <= T
                    t = idx;
                    k = phi{j}(t);
                    
                    n_kv(k,v) = n_kv(k,v) + 1;
                    n_k(k) = n_k(k) + 1;
                    n_jt{j}(t) = n_jt{j}(t) + 1;
                    n_jtv{j}(t,v) = n_jtv{j}(t,v) + 1;
                elseif T < idx && idx <= (T + K)
                    tspawned = tspawned + 1;
                    
                    M = M + 1;

                    k = idx - T;
                    
                    n_kv(k,v) = n_kv(k,v) + 1;
                    n_k(k) = n_k(k) + 1;
                    m_k(k) = m_k(k) + 1;
                    n_jt{j} = [n_jt{j}; 1];
                    
                    T = length(n_jt{j});
                    p = zeros(1 + T + K + 1, 1);
                    phi{j} = [phi{j}; k];
                    
                    t = T;
                    
                    n_jtv{j} = [n_jtv{j}; zeros(1,V)];
                    n_jtv{j}(t,v) = 1;
                elseif idx == (T + K + 1)
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
                    p = zeros(1 + T + K + 1, 1);
                    phi{j} = [phi{j}; k];
                    
                    t = T;
                    
                    n_jtv{j} = [n_jtv{j}; zeros(1,V)];
                    n_jtv{j}(t,v) = 1;
                else
                    assert(false, 'CRF: indexing is not working (table)');
                end
                
                topicMat{j}(n) = k;
                tableMat{j}(n) = t;
            end
            
            assert(K == length(m_k) && K == length(n_k), 'CRF: K and length is inconsistent');
            assert(isequal(n_k, sum(n_kv,2)), 'CRF: n_k and n_kv is inconsistent');
            assert(M == sum(m_k) && M == sum(cell2mat(cellfun(@(x) length(x), n_jt, 'UniformOutput', false)))...
                && M == sum(cell2mat(cellfun(@(x) length(x), phi, 'UniformOutput', false)))...
                ,'CRF: M and sum is inconsistent');
        end
        
        assert(sum(n_j) == sum(n_k) && sum(n_j) == sum(cell2mat(cellfun(@(x) sum(x), n_jt, 'UniformOutput', false)))...
             ,'CRF: n_j, n_k and n_jt is incosistent');
        
        figure(1);
        subplot(211); stem(m_k, 'MarkerSize', 0); title('m_k');
        subplot(212); stem(n_k, 'MarkerSize', 0); title('n_k');
        
        figure(2);
        imagesc(n_kv); caxis([0 max(max(n_kv))]); set(gca, 'XTick', []); title('n_k_v');
        
        assert(M == (M_ini + tspawned - tremoved), 'add/remove count of M is inoconsistent');
        assert(K == (K_ini + kspawned - kremoved), 'add/remove count of K is inoconsistent');
        fprintf('table_num_update: %d = %d + %d - %d\n', M, M_ini, tspawned, tremoved);
        fprintf('topic_num_update: %d = %d + %d - %d\n', K, K_ini, kspawned, kremoved);
        
        % Sampling topics
        K_ini = K;
        kspawned = 0;
        kremoved = 0;
        
        for j=1:J
            T = length(n_jt{j});
            p = zeros(1 + K + 1, 1);
            
            rnd = rand(T, 1);
            
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
                    
                    [K, n_k, n_kv, m_k, topicMat, phi] = deleteTopics(k, K, V, J, n_k, n_kv, m_k, topicMat, phi);
                    p = zeros(1 + K + 1, 1);
                end
                
                cnt = 1;
                
                for k=1:K
                    buf...
                        = log(m_k(k)) + loggamaprx(n_k(k) + betaV) - loggamaprx(n_k(k) + n_jt{j}(t) + betaV)...
                        + sum(log(gamma(n_kv(k,:) + n_jtv{j}(t,:) + beta))) - sum(log(gamma(n_kv(k,:) + beta)));
                    
                    p(cnt+1) = exp(buf) + p(cnt);
                    cnt = cnt + 1;
                end
                
                buf...
                    = log_gam + loggam_betaV - loggamaprx(n_jt{j}(t) + betaV)...
                    + sum(log(gamma(n_jtv{j}(t,:) + beta))) - Vloggam_beta;
                p(cnt+1) = exp(buf) + p(cnt);
                
                idx = find(p < p(end)*rnd(t));
                idx = idx(end);
                
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
                    
                    p = zeros(1 + K + 1, 1);
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
        end
        
        assert(sum(n_j) == sum(n_k) && sum(n_j) == sum(cell2mat(cellfun(@(x) sum(x), n_jt, 'UniformOutput', false)))...
             ,'CRF: n_j, n_k and n_jt is incosistent');
        assert(isequal(n_jt{j}, sum(n_jtv{j},2)), 'CRF: n_jt and n_jtv is incosistent');
        
        figure(1);
        subplot(211); stem(m_k, 'MarkerSize', 0); title('m_k');
        subplot(212); stem(n_k, 'MarkerSize', 0); title('n_k');
        
        figure(2);
        imagesc(n_kv); caxis([0 max(max(n_kv))]); set(gca, 'XTick', []); title('n_k_v');
        
        assert(K == (K_ini + kspawned - kremoved), 'add/remove count of K is inoconsistent');
        fprintf('topic_num_update: %d = %d + %d - %d\n\n', K, K_ini, kspawned, kremoved);
    end
end

%% deletion of topics
function [K, n_k, n_kv, m_k, topicMat, phi] = deleteTopics(k, K, V, J, n_k, n_kv, m_k, topicMat, phi)
    K = K - 1;

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
        phi(:,k) = dirichletrnd(n_kv(:,k) + beta);
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

%% count n_jk & n_k
function [n_jk, n_k] = count_jk(topicMat, J, K)
    n_jk = zeros(K, J);
    n_k  = zeros(K, 1);
    
    for k=1:K
        n_jk(k,:) = cell2mat(cellfun(@(x) length(find(x==k)), topicMat, 'UniformOutput', false))';
        
        topic_idx = cellfun(@(x) x == k, topicMat, 'UniformOutput', false);
        topic_idx = cellfun(@(x) x(:), topic_idx, 'UniformOutput', false);
        numk = cellfun(@(x) length(x(x)), topic_idx, 'UniformOutput', false);
        
        n_k(k) = sum([numk{:}]);
    end
end

%% count n_kv
function [n_kv] = count_kv(topicMat, w, K, V)
    n_kv = zeros(V, K);
    
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
            
            n_kv(v,k) = sum([numk{:}]);
            
            fprintf(repmat('\b',1,length(msg)));
        end

        fprintf('...\n');
    end
end

%% approximation of log-gamma function
% For example, compare result of the calculation below.
% A = loggamaprx(50);
% B = log(gamma(50));
function y = loggamaprx(x)
    y = 0.5*(log(2*pi) - log(x)) + x*(log(x + 1/(12*x - 1/(10*x))) - 1);
end