function [repository, pi] = directassignment(repository, alpha, gamma, beta, pi, init, maxitr)
    if ~exist('init','var')
        init = false;
    end
    
    if ~exist('maxitr','var')
        maxitr = 50;
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

    %% Main loop
    for itr=1:maxitr
    docidx = randperm(J);
    logp = zeros(K + 1, 1);

        %% Dish assignment
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

                    mu_0 = betarnd(1, gamma);
                    pi_new_0 = pi(1) * (1 - mu_0);
                    pi_new_K = pi(1) * mu_0;

                    pi_new = zeros(K+1,1);
                    pi_new(1) = pi_new_0;
                    pi_new(2:K) = pi(2:K);
                    pi_new(K+1) = pi_new_K;

                    pi = pi_new;

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
    
        %% Table sampling
        m_k = zeros(K, 1);
        for k=1:K
            str = sprintf(' (%d/%d)', k, K);
            strlen = length(str);
            fprintf(str);
            
            for j=1:J
                [m_jk, ~] = crp(alpha*pi(k+1), n_jk(k,j));
                m_k(k) = m_k(k) + m_jk;
            end
            
            str = repmat('\b', 1, strlen);
            fprintf(str);
        end
        
        M = sum(m_k);
    end
    
    %% Set to repository
    repository.topicMat = topicMat;
    
    repository.K = K;
    repository.M = M;
    
    repository.n_k      = n_k;
    repository.n_kv     = n_kv;
    repository.n_jk     = n_jk;
    repository.m_k      = m_k;
end