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
    beta    = 0.1;    % parameter of dirichlet distribution (symmetric)
    
    a_alpha = 1; % Shape parameter of gamma prior for alpha
    b_alpha = 1; % Scale parameter of gamma prior for alpha
    alpha   = 1;

    a_gam   = 2; % Shape parameter of gamma prior for gamma
    b_gam   = 4; % Scale parameter of gamma prior for gamma
    gam     = 0.5;
    
    %% Setup test condition
    maxitr = 100;
    L = zeros(N, maxitr);
    perplexity = zeros(maxitr, 1);
    
    %% CRF initialization
    %%{
    repository = crf(repository, alpha, gam, beta, true, 1);
    pi = dirichletrnd([gam; repository.m_k]);
    theta = ltopicrnd(pi, alpha, repository.n_jk, repository.J, repository.K);
    %}
    
    %% CRF inference
    vocabulary = repository.vocabulary;
    w          = repository.w;
    V          = repository.V;
    J          = repository.J;
    n_j        = repository.n_j;
    for i=1:20
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

        [gam, alpha] = hyprprmrnd(K, M, gam, alpha, n_j, a_gam, b_gam, a_alpha, b_alpha, 0);

        fprintf('-after-\n');
        K_mean = mean_k(gam, M);
        fprintf(' E(K) = %3.3f (gam = %3.3f, K = %d)\n', K_mean, gam, K);

        M_mean = mean_m(alpha, n_j);
        fprintf(' E(M) = %3.3f (alpha = %3.3f, M = %d)\n', M_mean, alpha, M);
    end
    
   %% Direct assignment inference initialization
   %{
   init = true;
   [repository, pi, theta] = directassignment(repository, alpha, beta, gam, 1, ones(J,1), init);
   %}
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