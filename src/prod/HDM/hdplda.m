function [perplexity, loglik, pi_best, z_best, phi_best] = hdplda(data, eta, alpha, beta, steps, maxitr)
    load(data);

    N = length(w);
    z = zeros(N, 1);
    z = ncrp(eta, z);

    K = 0;
    pi = sbptree(alpha, K + 1, z);

    myfun = @(n_kv, n_k) f_k(n_kv, n_k, beta, V);
    myworddist = @(K, n_kv) worddist(V, K, n_kv, beta);

    perplexity = zeros(maxitr, 1);
    loglik = zeros(N, maxitr);
    repo = [];
    bestscore = Inf;
    
    for itr=1:maxitr
        tic;
        [pi, repo] = directassignmentWrapper(w, alpha, pi, z, myfun, repo, steps);
        pi = hdpposteriorWrapper(pi, repo, alpha, z);
        [pi, z] = categoryassignment(eta, alpha, pi, z);
        
        K = length(pi{1}) - 1;
        phi = myworddist(K, repo.n_kv);
        
        [perplexity_tmp, loglik_tmp] = evalperp(w, pi{end}, phi);
        perplexity(itr) = perplexity_tmp;
        loglik(:, itr) = loglik_tmp;
        t = toc;
        
        if perplexity_tmp < bestscore
            pi_best = pi;
            z_best = z;
            phi_best = phi;
            bestscore = perplexity_tmp;
            bestitr = itr;
        end
        
        fprintf('Iteration %d: perplexity = %e (%e at %d), elapsed time = %3.3f\n', itr, perplexity_tmp, bestscore, bestitr, t);
        figure(1); 
        subplot(211); plot(perplexity(1:itr));
        subplot(212); imagesc(loglik(:, 1:itr)); colorbar
        drawnow;
    end
end