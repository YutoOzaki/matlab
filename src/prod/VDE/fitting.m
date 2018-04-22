function fitting(datafile, numepoch, K)
    %% load data
    load(datafile);
    
    %% load autoencoders and transform the data
    load('pretrained.mat');
    encnet = bestprms.encnet;
    encrpm = bestprms.encrpm;
    
    %% define prior net
    J = encrpm.reparam.J;
    L = 100;
    gam = 1;
    
    priornet = struct(...
        'reparam', reparamtrans(J, L),...
        'weight', mogtrans(K, J, gam, adagrad(5e-2, 1e-8, 'asc'))...
        );
    
    priornet.weight.init();
    init(data, encnet, encrpm, priornet);
    
    %% define configuration
    N = size(data, 2);
    batchsize = 100;
    numbatch = floor(N / batchsize);
    batchidx = zeros(2, 1);
    bestprms = struct('priornet', priornet);
    bestscore = -Inf;
    
    %% main loop
    loss_hist = zeros(numepoch);
    labels = zeros(N, 1);
    rprsn = zeros(J, N);
    
    for epoch=1:numepoch
        rndidx = randperm(N);
        c = rand(K, 3);
        c = bsxfun(@rdivide, c, sum(c));
        
        for batch=1:numbatch
            batchidx(1) = batchidx(2) + 1;
            batchidx(2) = batchidx(1) + batchsize - 1;
            x = data(:, rndidx(batchidx(1):batchidx(2)));
            
            names = fieldnames(encnet);
            input = x;
            for i=1:length(names)
                input = encnet.(names{i}).forwardprop(input);
            end

            mu = encrpm.mu.forwardprop(input);
            sigsq = encrpm.exp.forwardprop(encrpm.lnsigsq.forwardprop(input));

            % forward propagation
            priornet.reparam.init();
            z = priornet.reparam.forwardprop(struct('mu', mu, 'sig', sigsq));
            z = mean(z, 3);
            loglik = fprop(priornet, z);
            loss_hist(epoch) = loss_hist(epoch) + loglik;
            
            % backward propagation
            priornet = bprop(priornet, z);
            
            % gradient checking
            %gradcheck(priornet, z)
            
            % update
            update(struct('priornet', priornet));
            
            % clustering assignment
            gam = posterior(x, encnet, encrpm, priornet);
            [~,I] = max(gam);
            labels(rndidx(batchidx(1):batchidx(2)), 1) = I';
            
            rprsn(:, rndidx(batchidx(1):batchidx(2))) = z;
        end
        
        if loss_hist(epoch) > bestscore
            bestscore = loss_hist(epoch);
            fprintf('--current best score is updated to %e at epoch %d--\n', bestscore, epoch);
            bestprms = struct('priornet', priornet);
        end
        
        figure(1); plot(loss_hist(1:epoch)); drawnow;
        fprintf('epoch %d, loss: %e\n', epoch, loss_hist(epoch));
        
        figure(2);
        for k=1:K
            idx = labels == k;
            scatter(rprsn(1, idx), rprsn(2, idx), 20, c(k, :)); hold on
            scatter(priornet.weight.prms.eta_mu(1, k), priornet.weight.prms.eta_mu(2, k),...
                    20, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', [0 .7 .7], 'LineWidth', 2); hold on
            drawellip(priornet.weight.prms.eta_mu(:, k), diag(exp(priornet.weight.prms.eta_lnsig(:, k)))); hold on
        end
        hold off
        
        figure(3);
        for k=1:K
            idx = labels == k;
            scatter3(data(1, idx), data(2, idx), data(3, idx), 20, c(k, :)); hold on
        end
        hold off
        drawnow;
        
        batchidx = batchidx .* 0;
    end
    
    savemodel('fit.mat', bestprms)
end

function savemodel(filename, bestprms)
    netnames = fieldnames(bestprms);
    for i=1:length(netnames)
        nodenames = fieldnames(bestprms.(netnames{i}));
        
        for j=1:length(nodenames)
            bestprms.(netnames{i}).(nodenames{j}).refresh();
        end
    end
    
    save(filename, 'bestprms');
end

function gradcheck(priornet, z)
    prmnames = fieldnames(priornet.weight.prms);
    eps = 1e-6;
    f = zeros(2, 1);
    d = zeros(2, 1);
    
    for i=1:length(prmnames)
        prm = priornet.weight.prms.(prmnames{i});
        
        m = size(prm, 1);
        n = size(prm, 2);
        x = randi(m);
        y = randi(n);
        val = prm(x, y);
        
        prm(x, y) = val + eps;
        priornet.weight.prms.(prmnames{i}) = prm;
        f(1) = fprop(priornet, z);
        
        prm(x, y) = val - eps;
        priornet.weight.prms.(prmnames{i}) = prm;
        f(2) = fprop(priornet, z);
        
        d(1) = (f(1) - f(2))/(2*eps);
        d(2) = priornet.weight.grad.(prmnames{i})(x, y);
        re = abs(d(1) - d(2))/max(abs(d(1)), abs(d(2)));
        
        fprintf('grad check: %s %e %e %e\n', prmnames{i}, d(1), d(2), re);
        prm(x, y) = val;
        priornet.weight.prms.(prmnames{i}) = prm;
    end
end

function priornet = bprop(priornet, z)
    K = priornet.weight.K;
    mu = priornet.weight.prms.eta_mu;
    sigsq = exp(priornet.weight.prms.eta_lnsig);
    p = priornet.weight.prms.p;
    q = softmax(p);
    PI = (q + priornet.weight.gam)./(sum(q) + K*priornet.weight.gam);
    J = priornet.weight.J;
    
    batchsize = size(z, 2);

    pdf = zeros(K, batchsize);
    for k=1:K
        pdf(k, :) = mvnpdf(z', mu(:, k)', diag(sigsq(:, k)));
    end

    % p
    dLdPI = zeros(K, 1);
    A = sum(bsxfun(@times, pdf, PI));
    for k=1:K
        dLdPI(k, 1) = sum(pdf(k, :)./A);
    end

    dPIdq = zeros(K, K);
    dqdp = zeros(K, K);
    for i=1:K
        j = i;
        idx = setdiff(1:K, i);
        
        dPIdq(i, j) = sum(q(idx)) + (K - 1).*priornet.weight.gam;
        dqdp(i, j) =  exp(p(i))*sum(exp(p(idx)));

        for j=1:(K-1)
            dPIdq(i, idx(j)) = -q(i) - priornet.weight.gam;
            dqdp(i, idx(j)) = -exp(p(i) + p(idx(j)));
        end
    end
    dPIdq = dPIdq./(sum(q) + K*priornet.weight.gam)^2;
    dqdp = dqdp./sum(exp(p))^2;
    
    gp = (dLdPI'*dPIdq*dqdp)';
    
    % mu
    dLdmu = zeros(J, K);
    for k=1:K
        A = bsxfun(@minus, z, mu(:, k));
        B = bsxfun(@rdivide, A, sigsq(:, k));
        C = PI(k) .* pdf(k, :);
        D = C ./ sum(bsxfun(@times, pdf, PI));
        dLdmu(:, k) = sum(bsxfun(@times, B, D), 2);
    end
    
    % sig
    dLdsig = zeros(J, K);
    for k=1:K
        A = PI(k).*pdf(k, :);
        B = A./sum(bsxfun(@times, pdf, PI));
        
        C = bsxfun(@minus, z, mu(:, k)).^2;
        D = bsxfun(@minus, C, sigsq(:, k));
        E = bsxfun(@rdivide, D, 2.*sigsq(:, k).^2);
        
        dLdsig(:, k) = sum(bsxfun(@times, B, E), 2);
    end
    
    glnsig = dLdsig .* exp(priornet.weight.prms.eta_lnsig);
    
    priornet.weight.grad = struct(...
        'p', gp./batchsize,...
        'eta_mu', dLdmu./batchsize,...
        'eta_lnsig', glnsig./batchsize...
        );
end

function loglik = fprop(priornet, z)
    [~, r] = priornet.weight.forwardprop(z);
    
    batchsize = size(z, 2);
    loglik = sum(log(sum(r, 1)))./batchsize;
end

function gam = posterior(x, encnet, encrpm, priornet)
    names = fieldnames(encnet);
    input = x;
    for i=1:length(names)
        input = encnet.(names{i}).forwardprop(input);
    end

    mu = encrpm.mu.forwardprop(input);
    sigsq = encrpm.exp.forwardprop(encrpm.lnsigsq.forwardprop(input));
    
    priornet.reparam.init();
    z = priornet.reparam.forwardprop(struct('mu', mu, 'sig', sigsq));
    z = mean(z, 3);
    gam = priornet.weight.forwardprop(z);
end

function init(data, encnet, encrpm, priornet)
    N = size(data, 2);
    batchsize = 100;
    K = priornet.weight.K;
    
    for k=1:K
        rndidx = randperm(N, batchsize);
        input = data(:, rndidx);
        
        names = fieldnames(encnet);
        for i=1:length(names)
            input = encnet.(names{i}).forwardprop(input);
        end
        
        mu = encrpm.mu.forwardprop(input);
        sigsq = encrpm.exp.forwardprop(encrpm.lnsigsq.forwardprop(input));
        
        priornet.reparam.init();
        z = priornet.reparam.forwardprop(struct('mu', mu, 'sig', sigsq));
        z = mean(z, 3);
        
        priornet.weight.prms.eta_mu(:, k) = mean(z, 2);
        priornet.weight.prms.eta_lnsig(:, k) = log(diag(cov(z')));
    end    
end