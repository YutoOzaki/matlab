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