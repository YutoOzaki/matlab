function fitting_kmeans(datafile, K, L, numepoch, gpumode, dispcluster, displabeling)
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
    gam = 1;
    isdiag = true;
    
    priornet = struct(...
        'reparam', reparamtrans(encrpm.reparam.J, L, numbatch, gpumode),...
        'weight', mogtrans(K, encrpm.reparam.J, gam, rmsprop(0.9, 1e-2, 1e-8, 'asc'), isdiag, gpumode)...
        );
    
    priornet.weight.init();
    init(data, encnet, encrpm, priornet);
    
    %% initialization
    bestprms = struct('priornet', priornet);
    bestscore = [-Inf, -1];
    z = transform(data, encnet, encrpm, priornet);
    
    if gpumode
        loss_hist = zeros(numepoch, 1, 'single', 'gpuArray');
        sig = zeros(encrpm.reparam.J, encrpm.reparam.J, K, 'single', 'gpuArray');
        PI = zeros(K, 1, 'single', 'gpuArray');
        bestscore = gpuArray(cast(bestscore, 'single'));
    else
        loss_hist = zeros(numepoch, 1);
        sig = zeros(encrpm.reparam.J, encrpm.reparam.J, K);
        PI = zeros(K, 1);
    end
    
    %% main loop
    for epoch=1:numepoch
        [idx, C] = kmeans(z', K, 'MaxIter', 10000);
        
        mu = C';
        for k=1:K
            k_idx = find(idx == k);
            sig(:, :, k) = cov(z(:, k_idx)');
            PI(k) = length(k_idx);
        end
        PI = PI./N;

        loss_hist(epoch) = posterior(z, PI, mu, sig, isdiag);
        
        if loss_hist(epoch) > bestscore(1)
            priornet.weight.prms.p = log(PI) - log(sum(exp(PI)));
            priornet.weight.prms.eta_mu = mu;
            
            for k=1:K
                priornet.weight.prms.eta_lnsig(:, k) = log(diag(sig(:, :, k)));
            end
            
            bestscore(1) = loss_hist(epoch);
            bestscore(2) = epoch;
            fprintf('--current best score is updated to %e at epoch %d--\n', bestscore(1), epoch);
            bestprms = struct('priornet', priornet);
        end
        
        fprintf('epoch %d, loss: %e', epoch, loss_hist(epoch));
        figure(1); plot(loss_hist(1:epoch));
        figure(2); dispcluster(z, idx, epoch);
        figure(3); displabeling(data, idx);
        drawnow;
        
        t = toc;
        fprintf(' [elapsed time %3.3f, %s]\n', t, datestr(datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss')));
    end
    
    savemodel(strcat(datafile, '_fit.mat'), bestprms);
    
    %% result
    for k=1:K
        sig(:, :, k) = diag(exp(bestprms.priornet.weight.prms.eta_lnsig(:, k)));
    end
    [loss, q] = posterior(z, bestprms.priornet.weight.getPI(), bestprms.priornet.weight.prms.eta_mu, sig, isdiag);
    
    [~, labels] = max(q);
    figure(2); dispcluster(z, labels, -1); title(sprintf('best score %e (epoch %d, %e)', loss, bestscore(2), bestscore(1)));
    figure(3); displabeling(data, labels);
    drawnow;
end

function [loss, q] = posterior(z, PI, mu, sig, isdiag)
    N = size(z, 2);
    K = length(PI);
    batchsize = 512;
    
    if isdiag
        mvnfun = @diagmvnpdf;
    else
        mvnfun = @mvnpdf;
    end
    
    numbatch = floor(N / batchsize);
    remsize = rem(N, batchsize);
    batchidx = zeros(2, 1);
    
    r = zeros(K, batchsize);
    r_rem = zeros(K, remsize);
    q = zeros(K, N);
    loss = 0;
    if isa(z, 'gpuArray')
        r = gpuArray(cast(r, 'single'));
        r_rem = gpuArray(cast(r_rem, 'single'));
        q = gpuArray(cast(q, 'single'));
        loss = gpuArray(cast(loss, 'single'));
    end
    
    for batch=1:numbatch
        batchidx(1) = batchidx(2) + 1;
        batchidx(2) = batchidx(1) + batchsize - 1;

        input = z(:, batchidx(1):batchidx(2));
        for k=1:K
            r(k, :) = PI(k) .* mvnfun(input', mu(:, k)', sig(:, :, k));
        end
        q(:, batchidx(1):batchidx(2)) = bsxfun(@rdivide, r, sum(r));

        loglik = sum(log(sum(r, 1)))./batchsize;
        loss = loss + loglik;
    end
        
    batchidx(1) = batchidx(2) + 1;
    batchidx(2) = batchidx(1) +remsize - 1;

    input = z(:, batchidx(1):batchidx(2));
    for k=1:K
        r_rem(k, :) = PI(k) .* mvnfun(input', mu(:, k)', sig(:, :, k));
    end
    q(:, batchidx(1):batchidx(2)) = bsxfun(@rdivide, r_rem, sum(r_rem));

    loglik = sum(log(sum(r_rem, 1)))./batchsize;
    loss = loss + loglik;
end

function z = transform(data, encnet, encrpm, priornet)
    N = size(data, 2);
    if isa(data, 'gpuArray')
        z = zeros(encrpm.reparam.J, N, 'single', 'gpuArray');
    else
        z = zeros(encrpm.reparam.J, N);
    end
    
    numbatch = priornet.reparam.poolsize;
    batchsize = floor(N / numbatch);
    remsize = rem(N, batchsize);
    batchidx = zeros(2, 1);
    names = fieldnames(encnet);
    priornet.reparam.init();
     
    for n = 1:numbatch
        batchidx(1) = batchidx(2) + 1;
        batchidx(2) = batchidx(1) + batchsize - 1;
    
        x = data(:, batchidx(1):batchidx(2));
        
        input = x;
        for i=1:length(names)
            input = encnet.(names{i}).forwardprop(input);
        end

        mu = encrpm.mu.forwardprop(input);
        sigsq = encrpm.exp.forwardprop(encrpm.lnsigsq.forwardprop(input));

        z_tmp = priornet.reparam.forwardprop(struct('mu', mu, 'sig', sigsq));
        z(:, batchidx(1):batchidx(2)) = mean(z_tmp, 3);
    end
    
    batchidx(1) = batchidx(2) + 1;
    batchidx(2) = batchidx(1) +remsize - 1;
        
    x = data(:, batchidx(1):batchidx(2));

    input = x;
    for i=1:length(names)
        input = encnet.(names{i}).forwardprop(input);
    end

    mu = encrpm.mu.forwardprop(input);
    sigsq = encrpm.exp.forwardprop(encrpm.lnsigsq.forwardprop(input));

    z_tmp = priornet.reparam.forwardprop(struct('mu', mu, 'sig', sigsq));
    z(:, batchidx(1):batchidx(2)) = mean(z_tmp, 3);
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
            z = zeros(length(find(idx == 1)), 1);
        end
        
        scatter3(rprsn(1, idx), rprsn(2, idx), z, 20, c(k, :)); hold on 
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

function init(data, encnet, encrpm, priornet)
    N = size(data, 2);
    batchsize = 128;
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