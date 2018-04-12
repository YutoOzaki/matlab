function userscript
    %% load data
    load('testdata.mat');
    N = size(data, 2);
    D = size(data, 1);
    
    %% define model
    load('trained.mat');
    J = 8;
    numnode = [D; 40; J; 40; D];
    
    K = 4;
    p = rand(K, 1);
    
    eta_mu = zeros(J, K);
    eta_sig = zeros(J, K);
    for k=1:K
        eta_mu(:, k) = mvnrnd(zeros(1, J), diag(3.*ones(J, 1)), 1)';
        eta_sig(:, k) = mvnrnd(zeros(1, J), diag(ones(J, 1)), 1)';
    end
    
    L = 2;
    
    names = fieldnames(encnet);
    x = data(:, 1);
    for i=1:length(names)
        x = encnet.(names{i}).forwardprop(x);
    end
    M = size(x, 1);
    infnet = struct(...
        'mu', lineartrans(J, M, rmsprop(0.9, 1e-3, 1e-8)),...
        'sig', lineartrans(J, M, rmsprop(0.9, 1e-3, 1e-8)),...
        'exp', exptrans(),...
        'reparam', reparamtrans(J, L)...
        );
    
    
    gennet = struct(...
        'connect', lineartrans(numnode(4), J, rmsprop(0.9, 1e-3, 1e-8)),...
        'activate', tanhtrans(),...
        'mu', lineartrans(numnode(5), numnode(4), rmsprop(0.9, 1e-3, 1e-8)),...
        'sig', lineartrans(numnode(5), numnode(4), rmsprop(0.9, 1e-3, 1e-8)),...
        'exp', exptrans()...
        );
    
    priornet = struct(...
        'reparam', reparamtrans(numnode(3), 1),...
        'weight', mogtrans(eta_mu, eta_sig, p, rmsprop(0.9, 1e-3, 1e-8))...
        );
    
    lossnode = lossfun();
    
    names = fieldnames(encnet);
    for i=1:length(names)
        encnet.(names{i}).init();
    end
    
    names = fieldnames(decnet);
    for i=1:length(names)
        decnet.(names{i}).init();
    end
    
    names = fieldnames(priornet);
    for i=1:length(names)
        priornet.(names{i}).init();
    end
    
    %% define configuration
    numepoch = 100;
    batchsize = 50;
    numbatch = floor(N / batchsize);
    batchidx = zeros(2, 1);
    
    %% main loop
    loss = zeros(numepoch, 1);
    label = zeros(N, 1);
            
    for epoch=1:numepoch
        rndidx = randperm(N);
        
        for batch=1:numbatch
            batchidx(1) = batchidx(2) + 1;
            batchidx(2) = batchidx(1) + batchsize - 1;
            x = data(:, rndidx(batchidx(1):batchidx(2)));
            
            % forward propagation
            encnet.reparam.init();
            priornet.reparam.init();
            loss(epoch) = loss(epoch) + fprop(x, encnet, decnet, priornet, lossnode);
            
            % backward propagation
            bprop(encnet, decnet, priornet, lossnode);
            
            % gradient checking
            %gradcheck(x, encnet, decnet, priornet, lossnode);
            
            % update
            update(struct('encnet', encnet, 'decnet', decnet, 'priornet', priornet));
            
            % clustering assignment
            gam = posterior(x, encnet, priornet);
            [~,I] = max(gam);
            label(batchidx(1):batchidx(2), 1) = I';
        end
        
        figure(1); plot(loss(1:epoch)); drawnow;
        fprintf('epoch %d, loss: %e\n', epoch, loss(epoch));
        
        figure(2);
        c = rand(K, 3);
        for k=1:K
            idx = label == k;
            scatter(data(1, idx), data(2, idx), 20, c(k, :)./sum(c(k, :)));hold on
        end
        hold off
        drawnow;
        
        batchidx = batchidx .* 0;
    end
end

function gam = posterior(x, encnet, priornet)
    za = encnet.connect.forwardprop(x);
    zh = encnet.activate.forwardprop(za);
    zmu = encnet.mu.forwardprop(zh);
    lnzsig = encnet.sig.forwardprop(zh);
    zsig = encnet.exp.forwardprop(lnzsig);
    
    priornet.reparam.init();
    z = priornet.reparam.forwardprop(struct('mu', zmu, 'sig', zsig));
    gam = priornet.weight.forwardprop(z);
end