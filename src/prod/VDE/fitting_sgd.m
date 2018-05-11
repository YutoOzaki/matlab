function fitting_sgd(datafile, K, L, numepoch, gpumode, takeover, dispcluster, displabeling)
    if ~exist('dispcluster', 'var')
        dispcluster = @dispcluster3D;
    end
    if ~exist('displabeling', 'var')
        displabeling = @displabeling3D;
    end
    
    %% load data
    load(datafile);
    
    if gpumode
        data = gpuArray(cast(data, 'single'));
    end
    
    %% load autoencoders and transform the data
    load(strcat(datafile, '_pretrained.mat'));
    encnet = bestprms.encnet;
    encrpm = bestprms.encrpm;
    
    %% define configuration
    N = size(data, 2);
    batchsize = 128;
    numbatch = floor(N / batchsize);
    
    %% define prior net
    if takeover
        load(strcat(datafile, '_fit.mat'));
        priornet = bestprms.priornet;
        priornet.weight.setoptm(rmsprop(0.9, 1e-1, 1e-8, 'asc'));
    else
        J = encrpm.reparam.J;
        gam = 1;
        isdiag = true;

        priornet = struct(...
            'reparam', reparamtrans(J, L, numbatch, gpumode),...
            'weight', mogtrans(K, J, gam, rmsprop(0.9, 1e-2, 1e-8, 'asc'), isdiag, gpumode)...
            );

        priornet.weight.init();
        init(data, encnet, encrpm, priornet);
    end
    
    priornet.reparam.L = L;
    
    %% main loop
    loss_hist = zeros(numepoch, 1);
    labels = zeros(N, 1);
    rprsn = zeros(encrpm.reparam.J, N);
    bestscore = -Inf;
    batchidx = zeros(2, 1);
    
    if gpumode
        loss_hist = gpuArray(cast(loss_hist, 'single'));
        labels = gpuArray(cast(labels, 'single'));
        rprsn = gpuArray(cast(rprsn, 'single'));
        bestscore = gpuArray(cast(bestscore, 'single'));
    end
    
    for epoch=1:numepoch
        %profile on
        tic;
        rndidx = randperm(N);
        priornet.reparam.init();
        
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
            z = priornet.reparam.forwardprop(struct('mu', mu, 'sig', sigsq));
            z = mean(z, 3);
            loglik = fprop(priornet, z);
            loss_hist(epoch) = loss_hist(epoch) + loglik;
            assert(isnan(loss_hist(epoch)) == 0, 'NaN is occured!');
            
            % backward propagation
            priornet = bprop(priornet, z);
            
            % gradient checking
            %gradcheck(priornet, z); pause;
            
            % apply fisher information to take natural gradient
            %priornet.weight.grad.eta_mu = priornet.weight.grad.eta_mu.*exp(priornet.weight.prms.eta_lnsig);
            
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
            savemodel(strcat(datafile, '_fit.mat'), bestprms);
        end
        
        figure(1); plot(loss_hist(1:epoch)); drawnow;
        fprintf('epoch %d, loss: %e', epoch, loss_hist(epoch));
        
        figure(2);
        dispcluster(rprsn, labels, epoch);
        
        figure(3);
        displabeling(data, labels);
        
        drawnow;
        
        batchidx = batchidx .* 0;
        
        t = toc;
        fprintf(' [elapsed time %3.3f, %s]\n', t, datestr(datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss')));
        %profile off
        %profile viewer
    end
end

function dispcluster3D(rprsn, labels, epoch)
    elem= unique(labels);
    K = length(elem);
    
    hsv = [linspace(0.0, 0.9, K)', 0.8.*ones(K, 1), 0.75.*ones(K, 1)];
    c = hsv2rgb(hsv);
    
    for k=1:K
        idx = labels == elem(k);
        
        if size(rprsn, 1) > 2
            z = rprsn(3, idx);
        else
            z = ones(length(find(idx == 1)), 1);
        end
        
        scatter3(rprsn(1, idx), rprsn(2, idx), z, 20, c(k, :)); hold on
        %{
        scatter3(priornet.weight.prms.eta_mu(1, k), priornet.weight.prms.eta_mu(2, k),...
                priornet.weight.prms.eta_mu(3, k), 20,...
                'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', [0 .7 .7], 'LineWidth', 2); hold on
        drawellip(priornet.weight.prms.eta_mu(:, k), diag(exp(priornet.weight.prms.eta_lnsig(:, k)))); hold on
        %}
    end
    hold off
    title(epoch);
end

function displabeling3D(data, labels)
    elem= unique(labels);
    K = length(elem);
    
    hsv = [linspace(0.0, 0.9, K)', 0.8.*ones(K, 1), 0.75.*ones(K, 1)];
    c = hsv2rgb(hsv);
    
    for k=1:K
        idx = labels == elem(k);
        scatter3(data(1, idx), data(2, idx), data(3, idx), 20, c(k, :)); hold on
    end
    hold off
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
    f = zeros(2, 1, class(z));
    d = zeros(2, 1, class(z));
    
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
    PI = priornet.weight.getPI();
    J = priornet.weight.J;
    gam = priornet.weight.gam;
    SAFEDIV = priornet.weight.SAFEDIV;
    
    mvnfun = priornet.weight.mvnpdfwrapper;
    
    batchsize = size(z, 2);

    pdf = zeros(K, batchsize);
    dLdPI = zeros(K, 1);
    dPIdq = zeros(K, K);
    dqdp = zeros(K, K);
    dLdmu = zeros(J, K);
    dLdsig = zeros(J, K);
    
    if isa(z, 'gpuArray')
        pdf = gpuArray(cast(pdf, 'single'));
        dLdPI = gpuArray(cast(dLdPI, 'single'));
        dPIdq = gpuArray(cast(dPIdq, 'single'));
        dqdp = gpuArray(cast(dqdp, 'single'));
        dLdmu = gpuArray(cast(dLdmu, 'single'));
        dLdsig = gpuArray(cast(dLdsig, 'single'));
    end
    
    for k=1:K
        pdf(k, :) = mvnfun(z', mu(:, k)', diag(sigsq(:, k)));
    end

    % p
    A = sum(bsxfun(@times, pdf, PI) + SAFEDIV);
    for k=1:K
        dLdPI(k, 1) = sum(pdf(k, :)./A);
    end

    for i=1:K
        j = i;
        idx = setdiff(1:K, i);
        
        dPIdq(i, j) = sum(q(idx)) + (K - 1).*gam;
        dqdp(i, j) =  exp(p(i))*sum(exp(p(idx)));

        dPIdq(i, idx) = -q(i) - gam;
        dqdp(i, idx) = -exp(p(i) + p(idx));
    end
    dPIdq = dPIdq./(sum(q) + K*gam)^2;
    dqdp = dqdp./sum(exp(p))^2;
    
    gp = (dLdPI'*dPIdq*dqdp)';
    
    % mu
    for k=1:K
        A = bsxfun(@minus, z, mu(:, k));
        B = bsxfun(@rdivide, A, sigsq(:, k));
        C = PI(k) .* pdf(k, :);
        D = C ./ sum(bsxfun(@times, pdf, PI) + SAFEDIV);
        dLdmu(:, k) = sum(bsxfun(@times, B, D), 2);
    end
    
    % sig
    for k=1:K
        A = PI(k).*pdf(k, :);
        B = A./sum(bsxfun(@times, pdf, PI) + SAFEDIV);
        
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
    
    z = priornet.reparam.forwardprop(struct('mu', mu, 'sig', sigsq));
    z = mean(z, 3);
    gam = priornet.weight.forwardprop(z);
end

function init(data, encnet, encrpm, priornet)
    N = size(data, 2);
    batchsize = 512;
    K = priornet.weight.K;
    rndidx = randperm(N);
    batchidx = zeros(2, 1);
    priornet.reparam.init();
    
    for k=1:K
        batchidx(1) = batchidx(2) + 1;
        batchidx(2) = batchidx(1) + batchsize - 1;
        input = data(:, rndidx(batchidx(1):batchidx(2)));
        %rndidx = randperm(N, batchsize);
        %input = data(:, rndidx);
        
        names = fieldnames(encnet);
        for i=1:length(names)
            input = encnet.(names{i}).forwardprop(input);
        end
        
        mu = encrpm.mu.forwardprop(input);
        sigsq = encrpm.exp.forwardprop(encrpm.lnsigsq.forwardprop(input));
        
        z = priornet.reparam.forwardprop(struct('mu', mu, 'sig', sigsq));
        z = mean(z, 3);
        
        priornet.weight.prms.eta_mu(:, k) = mean(z, 2);
        priornet.weight.prms.eta_lnsig(:, k) = log(diag(cov(z')));
    end    
end